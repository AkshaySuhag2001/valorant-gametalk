# Local data files

Place the local training export here as training_rows.json. It should be a JSON list. Each row has a structured chat prompt and a structured assistant completion:

    {
      "prompt": [
        {"role": "system", "content": "Your system instruction"},
        {"role": "user", "content": "A question"}
      ],
      "completion": [
        {"role": "assistant", "content": "A concise answer"}
      ]
    }

The training file is ignored by Git and is not included in the public project package. The holdout question list is included separately for evaluation.
