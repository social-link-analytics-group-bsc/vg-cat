from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd
from rapidfuzz import fuzz
import torch
import numpy as np


class CatalanGrammarChecker:
    def __init__(self):

        token = "TUTOKEN"  # Insert your hugging face token here  
        self.model_name = "BSC-LT/roberta-base-ca"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=token)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name, token=token)
        
    def predict_masked_word(self, text, target_word):
        # Reemplazar la palabra objetivo con [MASK] en el texto
        masked_text = text.replace(target_word, self.tokenizer.mask_token, 1)
        inputs = self.tokenizer(masked_text, return_tensors="pt", truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Encontrar la posición del token [MASK]
        mask_token_index = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1]
        logits = outputs.logits[0, mask_token_index, :]
        top_tokens = torch.topk(logits, 5, dim=1).indices[0].tolist()
        top_words = [self.tokenizer.decode([token]).strip() for token in top_tokens]
        return top_words

    def choose_correct_option(self, option1, option2):
        # Encontrar la palabra que difiere entre las dos opciones
        words1 = option1.split()
        words2 = option2.split()
        
        # Si las frases tienen diferente número de palabras, usar perplexity
        if len(words1) != len(words2):
            score1 = self.evaluate_grammar(option1)
            score2 = self.evaluate_grammar(option2)
            return option1 if score1 < score2 else option2
        
        diff_indices = []
        for i, (w1, w2) in enumerate(zip(words1, words2)):
            if w1 != w2:
                diff_indices.append(i)
        
        # Si hay más de una diferencia, usar perplexity
        if len(diff_indices) != 1:
            score1 = self.evaluate_grammar(option1)
            score2 = self.evaluate_grammar(option2)
            return option1 if score1 < score2 else option2
        
        diff_index = diff_indices[0]
        target_word = words1[diff_index]
        context_text = ' '.join(words1)
        
        # Predecir la palabra enmascarada
        predicted_words = self.predict_masked_word(context_text, target_word)
        
        # Verificar cuál opción contiene la palabra predicha
        if words2[diff_index] in predicted_words:
            return option2
        elif words1[diff_index] in predicted_words:
            return option1
        else:
            # Si ninguna coincide, usar perplexity como fallback
            score1 = self.evaluate_grammar(option1)
            score2 = self.evaluate_grammar(option2)
            return option1 if score1 < score2 else option2

    def evaluate_grammar(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        loss = torch.nn.functional.cross_entropy(
            outputs.logits.view(-1, outputs.logits.size(-1)),
            inputs["input_ids"].view(-1),
            reduction="mean"
        )
        return torch.exp(loss).item()


class DataPreprocessor: 
    """
    """

    def remove_blank_spaces(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        """
        df.columns = [c.strip() for c in df.columns]
        for col in df.select_dtypes(include="object"): 
            df[col] = df[col].str.strip()
        return df

    def count_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        return f"Número de filas duplicadas: {df.duplicated().sum()}"    

    def normalize_typos(self, df: pd.DataFrame, threshold: int = 96) -> pd.DataFrame:
        """
        """

        grammar_checker = CatalanGrammarChecker()

        for col in df.columns:
   
            # Get unique normalized values and their frequencies
            unique_vals = df[col].value_counts().to_dict()
            unique_list = list(unique_vals.keys())

            # Find similar pairs above the threshold
            similar_pairs = []
            n = len(unique_list)
            for i in range(n):
                for j in range(i + 1, n):
                    similarity = fuzz.ratio(unique_list[i], unique_list[j])
                    if similarity >= 96:
                        similar_pairs.append((unique_list[i], unique_list[j], similarity))

            # Normalizar los errores
            for par in similar_pairs:
                option1, option2, similarity = par
                correct_option = grammar_checker.choose_correct_option(option1, option2)
               

                if correct_option == option1:
                    df[col] = df[col].astype(str).replace(option2, option1)
                else:
                    df[col] = df[col].astype(str).replace(option1, option2)

        return df

    def summarize_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        """
        cols = []
        for c in df.columns:
            n_missing = int(df[c].isna().sum())
            pct_missing = float(n_missing) / len(df) if len(df) > 0 else 0.0
            n_unique = int(df[c].nunique(dropna=True))
            dtype = str(df[c].dtype)
            cols.append((c, n_missing, pct_missing, n_unique, dtype))
        return pd.DataFrame(cols, columns=["column", "n_missing", "pct_missing", "n_unique", "dtype"])\
                .sort_values("pct_missing", ascending=False)


