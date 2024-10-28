import os
import json
import pickle
from tqdm import tqdm

BLINK_ROOT = f"{os.path.abspath(os.path.dirname(__file__))}/../.."

dates = ["230620", "230820", "231020", "231220", "240220", "240420"]

for date in dates:
    input_dir = os.path.join(BLINK_ROOT, "data", "tempcca", "documents", date)
    output_fpath = os.path.join(
        BLINK_ROOT, "data", "tempcca", "processed", date, "dictionary.pickle"
    )
    
    os.makedirs(os.path.dirname(output_fpath), exist_ok=True)
    
    dictionary = []
    label_ids = set()

    for doc_fname in tqdm(os.listdir(input_dir), desc=f"Loading documents for {date}"):
        assert doc_fname.endswith(".json")
        entity_type = doc_fname.split(".")[0]
        if entity_type in ["train", "test", "val"]:
            continue
        with open(os.path.join(input_dir, doc_fname), "r") as f:
            for idx, line in enumerate(f):
                record = {}
                entity = json.loads(line.strip())
                record["cui"] = entity["document_id"]
                record["title"] = entity["title"]
                record["description"] = entity["text"]
                record["type"] = entity_type
                dictionary.append(record)
                label_ids.add(record["cui"])

    assert len(dictionary) == len(label_ids)

    print(f"Finished reading {len(dictionary)} entities for {date}")
    print("Saving entity dictionary...")

    with open(output_fpath, "wb") as write_handle:
        pickle.dump(dictionary, write_handle, protocol=pickle.HIGHEST_PROTOCOL)
