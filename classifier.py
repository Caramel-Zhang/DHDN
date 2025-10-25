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

# Set non-interactive Matplotlib backend
plt.switch_backend('Agg')
plt.rcParams['figure.dpi'] = 100

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set random seeds for reproducibility
random.seed(4)
np.random.seed(4)
tf.random.set_seed(2)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

# GPU settings
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# Check GPU availability
print(tf.config.list_physical_devices('GPU'))

# Load dataset
dataset_name = 'ManySig'
dataset_path = './'
compact_dataset = load_compact_pkl_dataset(dataset_path, dataset_name)
tx_list = compact_dataset['tx_list']
rx_list = compact_dataset['rx_list']
capture_date_list = compact_dataset['capture_date_list']
n_rx = len(rx_list)
n_tx = len(tx_list)  # Train on all 6 classes
unknown_tx_idx = len(tx_list) - 1  # Last class is OOD
known_tx_list = tx_list[:-1]
print(f"Number of TX: {n_tx}, RX: {n_rx}, OOD TX: {tx_list[unknown_tx_idx]}")


# Function to add 0 dB Gaussian noise
def add_0db_noise(signals):
    """
    Add Gaussian noise to signals to achieve 0 dB SNR.
    signals: numpy array of shape (n_samples, 256, 2) [real, imag]
    Returns: signals with added noise
    """
    noisy_signals = signals.copy()
    n_samples = signals.shape[0]

    for i in range(n_samples):
        signal_std = np.std(signals[i])
        noise = np.random.normal(0, signal_std/ np.sqrt(10), signals[i].shape)
        noisy_signals[i] += noise

    return noisy_signals


# UMAP visualization function
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


# Create enhanced CNN model with residual connections
def create_cnn_model(n_tx):
    input_signal = layers.Input(shape=(256, 2), name='signal_input')
    x = layers.Permute((2, 1))(input_signal)
    x = layers.Reshape((512, 1))(x)

    shortcut_input = x

    x = layers.Conv1D(128, 3, strides=1, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    shortcut = layers.Conv1D(128, 1, strides=2, padding='same')(shortcut_input)
    shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Add()([x, shortcut])

    shortcut_input = x

    x = layers.Conv1D(32, 3, strides=1, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(32, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    shortcut = layers.Conv1D(32, 1, strides=2, padding='same')(shortcut_input)
    shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Add()([x, shortcut])

    x = layers.Conv1D(16, 3, strides=1, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(16, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling1D()(x)
    feature_layer = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01),
                                 name='feature_layer')(x)
    x = layers.Dropout(0.4)(feature_layer)
    output = layers.Dense(n_tx, activation='softmax', name='classification_output')(x)

    model = Model(inputs=input_signal, outputs=output)
    return model


# Custom CE + OE loss function
# Custom CE + EnergyOE loss function
def ce_oe_loss(y_true, y_pred):
    """
    Cross-Entropy loss for known classes +
    Energy-based OE loss for OOD samples.
    """
    # logits before softmax
    logits = cnn_model.get_layer("classification_output").output  # only works if we pass logits
    # but here we only get y_pred (softmax), so need trick:
    # => compute logits via inverse softmax (safe with log)
    eps = 1e-12
    logits = tf.math.log(tf.clip_by_value(y_pred, eps, 1.0))  # approximate logits

    # Known / OOD mask
    labels = tf.argmax(y_true, axis=1)
    is_known = tf.cast(tf.not_equal(labels, unknown_tx_idx), tf.float32)

    # Cross-entropy loss (only for known)
    ce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred) * is_known

    # Energy function
    T = 1.0
    energy = -T * tf.math.reduce_logsumexp(logits / T, axis=1)

    # Margin-based energy OE loss
    m_in, m_out = -1.0, -5.0   # margins (hyperparams)
    lambda_oe = 0.1

    energy_in = energy * is_known
    energy_out = energy * (1.0 - is_known)

    loss_in = tf.nn.relu(energy_in - m_in)
    loss_out = tf.nn.relu(m_out - energy_out)

    oe_loss = tf.reduce_mean(loss_in + loss_out)

    total_loss = tf.reduce_mean(ce_loss) + lambda_oe * oe_loss
    return total_loss



# Confusion matrix visualization function
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


# Main training and testing loop
TRAIN = True
nreal = 1
sig_len_list = [20]
patience = 15
n_epochs = 222
batch_size = 16
os.makedirs('weights1', exist_ok=True)
np.random.seed(9)

# Data preparation
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

# Add 0 dB noise to the signals
print("Adding 0 dB Gaussian noise to training, validation, and test sets...")
sig_train = add_0db_noise(sig_train)
sig_valid = add_0db_noise(sig_valid)
sig_smTest = add_0db_noise(sig_smTest)

# Verify number of classes
train_classes = np.unique(txidNum_train)
valid_classes = np.unique(txidNum_valid)
test_classes = np.unique(txidNum_smTest)
print(f"Number of unique classes in train: {len(train_classes)}")
print(f"Number of unique classes in valid: {len(valid_classes)}")
print(f"Number of unique classes in test: {len(test_classes)}")
print(f"Train classes: {train_classes}")
print(f"Valid classes: {valid_classes}")
print(f"Test classes: {test_classes}")

# Verify class distribution
print(f"Train class counts: {np.sum(txid_train, axis=0)}")
print(f"Valid class counts: {np.sum(txid_valid, axis=0)}")
print(f"Test class counts: {np.sum(txid_smTest, axis=0)}")

# Multiple trials
for tt in range(nreal):
    print(f"\n{'-' * 20}\nTrial: {tt + 1}\n{'-' * 20}")
    fname_w = f'weights1/cnn_trial_{tt + 1}.h5'

    # Create and train model
    cnn_model = create_cnn_model(n_tx)
    cnn_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss=ce_oe_loss,
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
            )
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

        # Save accuracy plot
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

        # Load best weights
        cnn_model.load_weights(fname_w)

    # Create feature extraction model
    feature_model = Model(
        inputs=cnn_model.input,
        outputs=cnn_model.get_layer('feature_layer').output
    )

    # Create model for closed-set evaluation (5 classes)
    closed_set_output = layers.Dense(n_tx - 1, activation='softmax', name='closed_set_output')(
        cnn_model.get_layer('feature_layer').output)
    closed_set_model = Model(inputs=cnn_model.input, outputs=closed_set_output)
    closed_set_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    # Copy weights from the main model
    for layer in closed_set_model.layers[:-1]:
        layer.set_weights(cnn_model.get_layer(layer.name).get_weights())

    # Test: Evaluate test set
    print(f"\nTesting on test set (Trial {tt + 1})...")
    test_features = feature_model.predict(sig_smTest, batch_size=batch_size, verbose=0)
    test_pred = cnn_model.predict(sig_smTest, batch_size=batch_size, verbose=0)
    test_pred_labels = np.argmax(test_pred, axis=1)
    test_labels = np.argmax(txid_smTest, axis=1)

    # Handle OOD class with confidence threshold
    confidence_threshold = 0.9
    test_pred_confidences = np.max(test_pred, axis=1)
    test_pred_labels_with_ood = test_pred_labels.copy()
    test_pred_labels_with_ood[test_pred_confidences < confidence_threshold] = unknown_tx_idx
    test_labels_with_ood = test_labels.copy()
    test_labels_with_ood[test_labels == unknown_tx_idx] = unknown_tx_idx

    # Evaluate accuracy on known classes only
    known_mask = test_labels != unknown_tx_idx
    if np.any(known_mask):
        txid_smTest_known = txid_smTest[known_mask][:, :n_tx - 1]  # Shape: (n_known_samples, 5)
        known_test_accuracy = \
        closed_set_model.evaluate(sig_smTest[known_mask], txid_smTest_known, batch_size=batch_size, verbose=0)[1]
        print(f"Test set accuracy (known classes only): {known_test_accuracy:.4f}")
    else:
        print("No known class samples in test set.")

    # Compute precision and recall for all classes
    precision = precision_score(test_labels_with_ood, test_pred_labels_with_ood, average=None)
    recall = recall_score(test_labels_with_ood, test_pred_labels_with_ood, average=None)
    for i, cls in enumerate(known_tx_list + ['OOD']):
        print(f"Class {cls} - Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}")

    # Save UMAP for test set
    visualize_umap(
        test_features, test_labels_with_ood,
        title=f"Test Set UMAP with OOD Class (Trial {tt})",
        filename=f'weights1/umap_trial_{tt }.png',
        classes=known_tx_list + ['OOD']
    )

    # Save confusion matrix for test set
    visualize_confusion_matrix(
        test_labels_with_ood, test_pred_labels_with_ood,
        classes=known_tx_list + ['OOD'],
        title=f"Confusion Matrix - Test Set with OOD Class (Trial {tt})",
        filename=f'weights1/confusion_trial_{tt }.png'
    )

    # Clear session to free memory
    tf.keras.backend.clear_session()