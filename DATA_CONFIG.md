### Dataset Configuration
The configuration of our dataset is as follows:

- Task 1. Entity Linking
  - {genre}_el.json
```
{ 
    post_id: {
        "timestamp": unix timestamp of post
        "post_url": url of post
        "body": text of post
        "date": date of post
        "annotations": [
            # annotated mention 1
            [
                mention name,
                start index,
                end index,
                corresponding wikipedia entity
            ],
            # annotated mention 2
            [
                mention name,
                ...
            ],
            ...
        ]
    },
    ...
}
```

- Task 2. Entity-Centric QA
  - {genre}_qa.json
```
{
    entity_name: {
        time_step: [
            # qa 1
            {
                "question": question
                "answer": answer
                "grounded_text": evidence text for the answer
                "retrieval_idx": [starting index, ending index] of grounded_text in the list of article sentences
                "mention_idx": [starting character index, ending character index] of mention in question
            },
            # qa 2
            ...
        ],
        ...
    },
    ...
}
```