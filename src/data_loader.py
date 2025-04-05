import re
from datasets import load_dataset
from transformers import AutoTokenizer
import config

def clean_html(text):
	"""Removes HTML tags from a string."""
	clean = re.compile('<.*?>')
	return re.sub(clean, '', text)

def load_and_preprocess_data(tokenizer_name=config.MODEL_NAME,
			dataset_name = config.DATASET_NAME,
			text_column = config.TEXT_COLUMN,
			label_column = config.LABEL_COLUMN,
			max_length = config.MAX_LENGTH,
			seed = config.SEED):
	"""
	Loads dataset, cleans the HTML format, removing </br> etc. 
	TOKENIZES the dataset, formats it

	Retuns:
		datasets.DatasetDict: Contains tokenized 'train' and 'test'
		splits.
	"""

	print(f"Loading dataset '{dataset_name}'...")
	raw_datasets = load_dataset(dataset_name)
	
	if 'unsupervised' in raw_datasets:
		raw_datasets.pop('unsupervised')
		print("Removed 'unsupervised' split.")

	print(f"Loading tokenizer '{tokenizer_name}'...")
	
	tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
	
	
	def preprocess_function(examples):
		
		cleaned_texts = [clean_html(text) for text in examples[text_column]]

		
		tokenized_inputs = tokenizer(cleaned_texts, 
						padding = 'max_length',
						truncation = True,
						max_length = max_length)
		return tokenized_inputs
	
	print("Preprocessing and tokenizing datasets...")

	tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
	
	tokenized_datasets = tokenized_datasets.remove_columns([text_column])
	tokenized_datasets = tokenized_datasets.rename_column(label_column, "labels")
	tokenized_datasets.set_format("torch")
	
	print("Dataset pre-processing complete...")
	print("\nSample processed data (first train exmaple:")
	print(tokenized_datasets["train"][0])

	return tokenized_datasets

if __name__ == "__main__":
	print(f"Running data loader directly using config: MAX_LENGTH = {config.MAX_LENGTH}")
	processed_data = load_and_preprocess_data()
	print("\nDataset structure after processing: ")
	print(processed_data)
	print(f"\nDevice specified in config: {config.DEVICE}")