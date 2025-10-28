import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, Model
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import seaborn as sns
import random
import umap.umap_ as umap
from data_utilities import *

plt.switch_backend('Agg')
plt.rcParams['figure.dpi'] = 100

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

random.seed(4)
np.random.seed(4)
tf.random.set_seed(2)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

print(tf.config.list_physical_devices('GPU'))

dataset_name = 'ManySig'
dataset_path = './'
compact_dataset = load_compact_pkl_dataset(dataset_path, dataset_name)
tx_list = compact_dataset['tx_list']
rx_list = compact_dataset['rx_list']
capture_date_list = compact_dataset['capture_date_list']
n_rx = len(rx_list)
n_tx = len(tx_list)
unknown_tx_idx = len(tx_list) - 1
known_tx_list = tx_list[:-1]
n_id_classes = len(known_tx_list)
print(f"Number of TX: {n_tx}, RX: {n_rx}, OOD TX: {tx_list[unknown_tx_idx]}")

prototypes = [tf.Variable(tf.random.normal([128], stddev=0.01), trainable=False, name=f'proto_{i}') for i in range(n_id_classes)]
mean_energies = [tf.Variable(tf.zeros([]), trainable=False, name=f'mu_{i}') for i in range(n_id_classes)]
alpha = tf.Variable(0.5, trainable=True, name='alpha')
momentum = 0.9

def add_0db_noise(signals):
    noisy_signals = signals.copy()
    n_samples = signals.shape[0]

    for i in range(n_samples):
        signal_std = np.std(signals[i])
        noise = np.random.normal(0, signal_std/ np.sqrt(10), signals[i].shape)
        noisy_signals[i] += noise

    return noisy_signals

class ChannelWiseShrinkage(layers.Layer):
    def __init__(self, **kwargs):
        super(ChannelWiseShrinkage, self).__init__(**kwargs)

    def build(self, input_shape):
        self.thresholds = self.add_weight(
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True,
            name='thresholds'
        )
        super(ChannelWiseShrinkage, self).build(input_shape)

    def call(self, inputs):
        abs_x = tf.abs(inputs)
        threshold = tf.nn.softplus(self.thresholds)
        mask = tf.cast(abs_x > threshold, tf.float32)
        shrunk = inputs * (1.0 - threshold / (abs_x + 1e-8)) * mask
        return shrunk

def rsbu_cw(filters, kernel_size=3, strides=1, projection=True):
    def rsbu_cw_block(inputs):
        shortcut = inputs

        x = layers.Conv1D(filters, kernel_size, strides=strides, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv1D(filters, kernel_size, strides=1, padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = ChannelWiseShrinkage()(x)

        if projection and strides != 1:
            shortcut = layers.Conv1D(filters, 1, strides=strides, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        elif inputs.shape[-1] != filters:
            shortcut = layers.Conv1D(filters, 1, strides=1, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        x = layers.Add()([x, shortcut])
        x = layers.ReLU()(x)

        return x
    return rsbu_cw_block

def create_cnn_model(n_tx):
    input_signal = layers.Input(shape=(256, 2), name='signal_input')
    x = layers.Reshape((512, 1))(input_signal)

    x = layers.Conv1D(64, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = rsbu_cw(16, strides=2)(x)
    x = rsbu_cw(32, strides=1)(x)
    x = rsbu_cw(64, strides=2)(x)
    x = rsbu_cw(128, strides=1)(x)

    x = layers.GlobalAveragePooling1D()(x)
    feature_layer = layers.Lambda(lambda t: tf.expand_dims(t, axis=-1), name='feature_layer')(x)

    output = layers.Dense(n_tx, activation='softmax', name='classification_output')(x)

    model = Model(inputs=input_signal, outputs=output)
    return model

class AdaptiveEnergyLoss(keras.losses.Loss):
    def __init__(self, lambda_ae=0.5, **kwargs):
        super().__init__(**kwargs)
        self.lambda_ae = lambda_ae
        self.ce_loss = keras.losses.CategoricalCrossentropy()

    def call(self, y_true, y_pred):
        logits = tf.math.log(tf.clip_by_value(y_pred, 1e-12, 1.0 - 1e-12))

        labels = tf.argmax(y_true, axis=1)
        is_id = tf.cast(tf.not_equal(labels, unknown_tx_idx), tf.float32)
        is_ood = 1.0 - is_id

        ce = self.ce_loss(y_true, y_pred)

        energy = -tf.math.reduce_logsumexp(logits, axis=1)

        print("Warning: Full adaptive prototype update requires custom training loop. Using fixed margins as fallback.")
        m_in = -1.0
        m_out = -5.0
        ae_loss = tf.reduce_mean(tf.nn.relu(energy * is_id - m_in) + tf.nn.relu(m_out - energy * is_ood))

        total_loss = ce + self.lambda_ae * ae_loss
        return total_loss

class EMAUpdateCallback(keras.callbacks.Callback):
    def __init__(self, feature_model, prototypes, mean_energies, momentum=0.9):
        super().__init__()
        self.feature_model = feature_model
        self.prototypes = prototypes
        self.mean_energies = mean_energies
        self.momentum = momentum

    def on_batch_end(self, batch, logs=None):
        pass

def visualize_umap(features, labels, title="UMAP Visualization", filename=None, classes=None):
    try:
        if np.isnan(features).any() or np.isinf(features).any():
            print("Warning: Input features contain NaN or Inf values. Cleaning data...")
            features = np.nan_to_num(features, nan=0.0, posinf=1e5, neginf=-1e5)
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std[std == 0] = 1.0
        features = (features - mean) / std
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        embedding = reducer.fit_transform(features)
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral',
                              s=15, alpha=0.7, edgecolors='w', linewidth=0.3)
        cbar = plt.colorbar(scatter, pad=0.01)
        cbar.set_ticks(np.arange(len(np.unique(labels))))
        cbar.set_ticklabels(classes if classes else np.unique(labels))
        plt.title(f"{title}\n(Points: {len(features)}, Classes: {len(np.unique(labels))})")
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        if filename:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename)
            print(f"Saved UMAP plot to {filename}")
            plt.close()
        else:
            plt.close()
    except Exception as e:
        print(f"Error saving UMAP plot to {filename}: {e}")
        plt.close()

def visualize_confusion_matrix(y_true, y_pred, classes, title="Confusion Matrix", filename=None):
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        if filename:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename)
            print(f"Saved confusion matrix to {filename}")
            plt.close()
        else:
            plt.close()
    except Exception as e:
        print(f"Error saving confusion matrix to {filename}: {e}")
        plt.close()

TRAIN = True
nreal = 1
sig_len_list = [20]
patience = 15
n_epochs = 222
batch_size = 16
os.makedirs('weights1', exist_ok=True)
np.random.seed(9)

dataset = merge_compact_dataset(compact_dataset, capture_date_list,
                                tx_list, rx_list,
                                max_sig=sig_len_list[0] + 100, equalized=0)
val_frac = 100 / (sig_len_list[0] + 200)
test_frac = 100 / (sig_len_list[0] + 200)

train_augset, val_augset, test_augset_smRx = prepare_dataset(
    dataset, tx_list, val_frac=0.1, test_frac=0.2)

[sig_train, txidNum_train, txid_train, cls_weights] = train_augset
[sig_valid, txidNum_valid, txid_valid, _] = val_augset
[sig_smTest, txidNum_smTest, txid_smTest, _] = test_augset_smRx

print("Adding 0 dB Gaussian noise to training, validation, and test sets...")
sig_train = add_0db_noise(sig_train)
sig_valid = add_0db_noise(sig_valid)
sig_smTest = add_0db_noise(sig_smTest)

train_classes = np.unique(txidNum_train)
valid_classes = np.unique(txidNum_valid)
test_classes = np.unique(txidNum_smTest)
print(f"Number of unique classes in train: {len(train_classes)}")
print(f"Number of unique classes in valid: {len(valid_classes)}")
print(f"Number of unique classes in test: {len(test_classes)}")
print(f"Train classes: {train_classes}")
print(f"Valid classes: {valid_classes}")
print(f"Test classes: {test_classes}")

print(f"Train class counts: {np.sum(txid_train, axis=0)}")
print(f"Valid class counts: {np.sum(txid_valid, axis=0)}")
print(f"Test class counts: {np.sum(txid_smTest, axis=0)}")

for tt in range(nreal):
    print(f"\n{'-' * 20}\nTrial: {tt + 1}\n{'-' * 20}")
    fname_w = f'weights1/cnn_trial_{tt + 1}.h5'

    cnn_model = create_cnn_model(n_tx)
    loss_fn = AdaptiveEnergyLoss(lambda_ae=0.5)
    cnn_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss=loss_fn,
        metrics=['categorical_accuracy']
    )

    if TRAIN:
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                fname_w,
                monitor='val_categorical_accuracy',
                save_best_only=True,
                mode='max'
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_categorical_accuracy',
                patience=patience,
                mode='max'
            ),
        ]

        history = cnn_model.fit(
            sig_train,
            txid_train,
            validation_data=(sig_valid, txid_valid),
            epochs=n_epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        try:
            plt.plot(history.history['categorical_accuracy'], label='Train Accuracy')
            plt.plot(history.history['val_categorical_accuracy'], label='Val Accuracy')
            plt.title(f'Classification Accuracy (Trial {tt + 1})')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            accuracy_plot_file = f'weights1/accuracy_curve_trial_{tt + 1}.png'
            os.makedirs(os.path.dirname(accuracy_plot_file), exist_ok=True)
            plt.savefig(accuracy_plot_file)
            print(f"Saved accuracy plot to {accuracy_plot_file}")
            plt.close()
        except Exception as e:
            print(f"Error saving accuracy plot to {accuracy_plot_file}: {e}")
            plt.close()

        cnn_model.load_weights(fname_w)

    feature_model = Model(
        inputs=cnn_model.input,
        outputs=cnn_model.get_layer('feature_layer').output[:, :, 0]
    )

    closed_set_features = cnn_model.get_layer('feature_layer').output[:, :, 0]
    closed_set_output = layers.Dense(n_id_classes, activation='softmax', name='closed_set_output')(closed_set_features)
    closed_set_model = Model(inputs=cnn_model.input, outputs=closed_set_output)
    closed_set_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    for layer in closed_set_model.layers[:-1]:
        if layer.name in [l.name for l in cnn_model.layers]:
            layer.set_weights(cnn_model.get_layer(layer.name).get_weights())

    print(f"\nTesting on test set (Trial {tt + 1})...")
    test_features = feature_model.predict(sig_smTest, batch_size=batch_size, verbose=0)
    test_pred = cnn_model.predict(sig_smTest, batch_size=batch_size, verbose=0)
    test_pred_labels = np.argmax(test_pred, axis=1)
    test_labels = np.argmax(txid_smTest, axis=1)

    confidence_threshold = 0.9
    test_pred_confidences = np.max(test_pred, axis=1)
    test_pred_labels_with_ood = test_pred_labels.copy()
    test_pred_labels_with_ood[test_pred_confidences < confidence_threshold] = unknown_tx_idx
    test_labels_with_ood = test_labels.copy()
    test_labels_with_ood[test_labels == unknown_tx_idx] = unknown_tx_idx

    known_mask = test_labels != unknown_tx_idx
    if np.any(known_mask):
        txid_smTest_known = np.zeros((np.sum(known_mask), n_id_classes))
        for i, true_label in enumerate(test_labels[known_mask]):
            if true_label < unknown_tx_idx:
                txid_smTest_known[i, true_label] = 1
        known_test_accuracy = \
        closed_set_model.evaluate(sig_smTest[known_mask], txid_smTest_known, batch_size=batch_size, verbose=0)[1]
        print(f"Test set accuracy (known classes only): {known_test_accuracy:.4f}")
    else:
        print("No known class samples in test set.")

    precision = precision_score(test_labels_with_ood, test_pred_labels_with_ood, average=None, zero_division=0)
    recall = recall_score(test_labels_with_ood, test_pred_labels_with_ood, average=None, zero_division=0)
    for i, cls in enumerate(known_tx_list + ['OOD']):
        print(f"Class {cls} - Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}")

    visualize_umap(
        test_features, test_labels_with_ood,
        title=f"Test Set UMAP with OOD Class (Trial {tt})",
        filename=f'weights1/umap_trial_{tt }.png',
        classes=known_tx_list + ['OOD']
    )

    visualize_confusion_matrix(
        test_labels_with_ood, test_pred_labels_with_ood,
        classes=known_tx_list + ['OOD'],
        title=f"Confusion Matrix - Test Set with OOD Class (Trial {tt})",
        filename=f'weights1/confusion_trial_{tt }.png'
    )

    tf.keras.backend.clear_session()