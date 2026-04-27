from typing import Any, Dict, List, Union
from transformers import DataCollatorForLanguageModeling

class DataCollatorForTwoLastTokensLM(DataCollatorForLanguageModeling):
    """Data Collator to make model only focus on the last two tokens."""
    
    def __init__(
        self,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs
    ):
        super().__init__(*args, mlm, *kwargs)
        self.ignore_index = ignore_index
    
    def torch_call(
        self, 
        examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        batch = super().torch_call(examples)
        
        for i in range(len(examples)):
            first_last_token_ids = (
                (batch["labels"][i] != self.ignore_index)
                .nonzero()[-2]
                .item()
            )
            
            # Ignore all tokens except for the last two tokens
            batch["labels"][i, :first_last_token_ids] = self.ignore_index
            
        return batch