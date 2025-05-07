import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

class CodeRedundancyPredictor:
    def __init__(self, model_path, tokenizer_path):
        """
        Initialize the predictor with trained model and tokenizers
        
        Args:
            model_path: Path to the saved model file (.keras)
            tokenizer_path: Path to the saved tokenizers file (.json)
        """
        def abs_diff(tensors):
            x, y = tensors
            return tf.abs(x - y)

        def mul_prod(tensors):
            x, y = tensors
            return x * y

        class PositionalEncoding(tf.keras.layers.Layer):
            def __init__(self, position, d_model):
                super(PositionalEncoding, self).__init__()
                self.position = position
                self.d_model = d_model
                
            def get_angles(self, pos, i, d_model):
                angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
                return pos * angle_rates
                
            def call(self, inputs):
                angle_rads = self.get_angles(np.arange(self.position)[:, np.newaxis],
                                           np.arange(self.d_model)[np.newaxis, :],
                                           self.d_model)
                
                angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
                angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
                
                pos_encoding = angle_rads[np.newaxis, ...]
                return inputs + tf.cast(pos_encoding, dtype=tf.float32)

        # Custom objects dictionary - must include ALL custom objects used in the model
        custom_objects = {
            'abs_diff': abs_diff,
            'mul_prod': mul_prod,
            'PositionalEncoding': PositionalEncoding,
            'Functional': tf.keras.models.Model  
        }

        try:
            print(f"Loading model from {model_path}")
            self.model = tf.keras.models.load_model(
                model_path,
                custom_objects=custom_objects,
                compile=False  # We'll compile manually to avoid metric issues
            )
            
            # Manually compile with the same configuration as training
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=[
                    'accuracy',
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                    tf.keras.metrics.AUC(name='auc')
                ]
            )
            print("Model loaded and compiled successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

        # Load tokenizers
        print(f"Loading tokenizers from {tokenizer_path}")
        try:
            with open(tokenizer_path, 'r') as f:
                tokenizer_data = json.load(f)
                
            self.code1_tokenizer = tokenizer_from_json(tokenizer_data['code1_tokenizer'])
            self.code2_tokenizer = tokenizer_from_json(tokenizer_data['code2_tokenizer'])
            
            # Load model parameters (should match training)
            self.max_length = 150
            self.vocab_size = 10000
            self.embedding_dim = 100
            
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizers: {str(e)}")

        print("Model and tokenizers loaded successfully!")

    def predict(self, code1, code2):
        """
        Predict if two code snippets are redundant
        
        Args:
            code1: First code snippet (string)
            code2: Second code snippet (string)
            
        Returns:
            float: Redundancy score between 0 and 1
            bool: True if the code is redundant (score > 0.6), False otherwise
        """
        # Preprocess input
        code1_seq = self.code1_tokenizer.texts_to_sequences([code1])
        code2_seq = self.code2_tokenizer.texts_to_sequences([code2])
        
        code1_padded = pad_sequences(code1_seq, maxlen=self.max_length, padding='post', truncating='post')
        code2_padded = pad_sequences(code2_seq, maxlen=self.max_length, padding='post', truncating='post')
        
        # Predict
        prediction = self.model.predict([code1_padded, code2_padded], verbose=0)
        prediction_value = prediction[0][0]
        is_redundant = prediction_value > 0.6
        
        return prediction_value, is_redundant
    
    def predict_from_files(self, file1_path, file2_path):
        """
        Predict redundancy between two code files
        
        Args:
            file1_path: Path to first code file
            file2_path: Path to second code file
            
        Returns:
            float: Redundancy score between 0 and 1
            bool: True if the code is redundant (score > 0.6), False otherwise
        """
        # Read files
        try:
            with open(file1_path, 'r', encoding='utf-8') as f:
                code1 = f.read()
                
            with open(file2_path, 'r', encoding='utf-8') as f:
                code2 = f.read()
                
            return self.predict(code1, code2)
            
        except Exception as e:
            print(f"Error reading files: {e}")
            return None, None

    def predict_batch(self, code_pairs):
        """
        Predict redundancy for multiple code pairs at once
        
        Args:
            code_pairs: List of tuples containing (code1, code2) pairs
            
        Returns:
            List of tuples with (redundancy_score, is_redundant) for each pair
        """
        # Preprocess all inputs
        code1_list = [pair[0] for pair in code_pairs]
        code2_list = [pair[1] for pair in code_pairs]
        
        code1_sequences = self.code1_tokenizer.texts_to_sequences(code1_list)
        code2_sequences = self.code2_tokenizer.texts_to_sequences(code2_list)
        
        code1_padded = pad_sequences(code1_sequences, maxlen=self.max_length, padding='post', truncating='post')
        code2_padded = pad_sequences(code2_sequences, maxlen=self.max_length, padding='post', truncating='post')
        
        # Batch predict
        predictions = self.model.predict([code1_padded, code2_padded], verbose=0).flatten()
        is_redundant = predictions > 0.6
        
        return list(zip(predictions, is_redundant))


    def predict(self, code1, code2):
        """
        Predict if two code snippets are redundant
        
        Args:
            code1: First code snippet (string)
            code2: Second code snippet (string)
            
        Returns:
            float: Redundancy score between 0 and 1
            bool: True if the code is redundant (score > 0.6), False otherwise
        """
        # Preprocess input
        code1_seq = self.code1_tokenizer.texts_to_sequences([code1])
        code2_seq = self.code2_tokenizer.texts_to_sequences([code2])
        
        code1_padded = pad_sequences(code1_seq, maxlen=self.max_length, padding='post', truncating='post')
        code2_padded = pad_sequences(code2_seq, maxlen=self.max_length, padding='post', truncating='post')
        
        # Predict
        prediction = self.model.predict([code1_padded, code2_padded], verbose=0)
        prediction_value = prediction[0][0]
        is_redundant = prediction_value > 0.6
        
        return prediction_value, is_redundant
    
    def predict_from_files(self, file1_path, file2_path):
        """
        Predict redundancy between two code files
        
        Args:
            file1_path: Path to first code file
            file2_path: Path to second code file
            
        Returns:
            float: Redundancy score between 0 and 1
            bool: True if the code is redundant (score > 0.6), False otherwise
        """
        # Read files
        try:
            with open(file1_path, 'r', encoding='utf-8') as f:
                code1 = f.read()
                
            with open(file2_path, 'r', encoding='utf-8') as f:
                code2 = f.read()
                
            return self.predict(code1, code2)
            
        except Exception as e:
            print(f"Error reading files: {e}")
            return None, None

    def predict_batch(self, code_pairs):
        """
        Predict redundancy for multiple code pairs at once
        
        Args:
            code_pairs: List of tuples containing (code1, code2) pairs
            
        Returns:
            List of tuples with (redundancy_score, is_redundant) for each pair
        """
        # Preprocess all inputs
        code1_list = [pair[0] for pair in code_pairs]
        code2_list = [pair[1] for pair in code_pairs]
        
        code1_sequences = self.code1_tokenizer.texts_to_sequences(code1_list)
        code2_sequences = self.code2_tokenizer.texts_to_sequences(code2_list)
        
        code1_padded = pad_sequences(code1_sequences, maxlen=self.max_length, padding='post', truncating='post')
        code2_padded = pad_sequences(code2_sequences, maxlen=self.max_length, padding='post', truncating='post')
        
        # Batch predict
        predictions = self.model.predict([code1_padded, code2_padded], verbose=0).flatten()
        is_redundant = predictions > 0.6
        
        return list(zip(predictions, is_redundant))


def main():
    # Path to saved model and tokenizers
    model_path = "C:/Users/piyus/Desktop/Major Project/Identification of Redundant Code Using AI/Sem 8/LSTM_Model/best_lstm_code_redundancy_model.keras"
    tokenizer_path = "C:/Users/piyus/Desktop/Major Project/Identification of Redundant Code Using AI/Sem 8/LSTM_Model/lstm_code_redundancy_tokenizers.json"
    
    # Initialize the predictor
    try:
        predictor = CodeRedundancyPredictor(model_path, tokenizer_path)
    except Exception as e:
        print(f"Failed to initialize predictor: {str(e)}")
        import sys
        sys.exit(1)
    
    while True:
        print("\nCode Redundancy Detector")
        print("1. Check two code snippets")
        print("2. Check two code files")
        print("3. Exit")
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            print("\nEnter first code snippet (type 'END' on a new line when finished):")
            code1_lines = []
            while True:
                line = input()
                if line == 'END':
                    break
                code1_lines.append(line)
            code1 = '\n'.join(code1_lines)
            
            print("\nEnter second code snippet (type 'END' on a new line when finished):")
            code2_lines = []
            while True:
                line = input()
                if line == 'END':
                    break
                code2_lines.append(line)
            code2 = '\n'.join(code2_lines)
            
            score, is_redundant = predictor.predict(code1, code2)
            print(f"\nRedundancy Score: {score:.4f}")
            print(f"Conclusion: The code snippets are {'redundant' if is_redundant else 'not redundant'}.")
            
        elif choice == '2':
            file1_path = input("\nEnter path to first code file: ")
            file2_path = input("Enter path to second code file: ")
            
            score, is_redundant = predictor.predict_from_files(file1_path, file2_path)
            if score is not None:
                print(f"\nRedundancy Score: {score:.4f}")
                print(f"Conclusion: The code files are {'redundant' if is_redundant else 'not redundant'}.")
            
        elif choice == '3':
            print("Exiting...")
            break
            
        else:
            print("Invalid choice. Please try again.")


# Example usage in a script without interactive menu
def example_usage():
    # Path to your saved model and tokenizers
    model_path = "C:/Users/piyus/Desktop/Major Project/Identification of Redundant Code Using AI/Sem 8/LSTM_Model/best_lstm_code_redundancy_model.keras"
    tokenizer_path = "C:/Users/piyus/Desktop/Major Project/Identification of Redundant Code Using AI/Sem 8/LSTM_Model/lstm_code_redundancy_tokenizers.json"
    
    # Initialize the predictor
    try:
        predictor = CodeRedundancyPredictor(model_path, tokenizer_path)
    
        # Example code snippets
        code1 = """
        int factorial(int n) {
            if (n <= 1) return 1;
            return n * factorial(n-1);
        }
        """
        
        code2 = """
        int compute_factorial(int x) {
            if (x <= 1) return 1;
            else return x * compute_factorial(x-1);
        }
        """
        
        # Make prediction
        score, is_redundant = predictor.predict(code1, code2)
        print(f"Redundancy Score: {score:.4f}")
        print(f"The code snippets are {'redundant' if is_redundant else 'not redundant'}.")
        
        # Process multiple pairs at once
        pairs = [
            (code1, code2),
            ("function add(a, b) { return a + b; }", "function sum(x, y) { return x + y; }"),
            ("def hello(): print('hello')", "function greet() { console.log('hello'); }")
        ]
        
        results = predictor.predict_batch(pairs)
        for i, (score, is_redundant) in enumerate(results):
            print(f"\nPair {i+1}:")
            print(f"Redundancy Score: {score:.4f}")
            print(f"Is Redundant: {is_redundant}")
    
    except Exception as e:
        print(f"Error in example usage: {str(e)}")


if __name__ == "__main__":
    # Use either the interactive menu or the example usage
    main()  # Interactive menu
    # example_usage()  # Non-interactive example