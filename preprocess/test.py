from preprocess import preprocess_folder
from eda import perform_eda
import os

def main():
  current_dir = os.path.dirname(os.path.abspath(__file__))

  data_folder = os.path.join(current_dir, "../data")

  preprocess_folder(data_folder,"text",data_folder+"/preprocessed")

  perform_eda(data_folder+"/preprocessed/combined_preprocessed.csv","cleaned_text")

if __name__ == "__main__":
  main()