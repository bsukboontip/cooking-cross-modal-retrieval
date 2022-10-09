# cooking-cross-modal-retrieval

## Updated Dataloader running

1. Create a folder `recipe1M_layers` outside the codebase folder
2. Add the data (`val`/`test`) folder outside the codebase folder
3. Download and add `cleaned_ingredients.json` and `cleaned_layers.json` files inside `recipe1M_layers` folder
4. Run and use dataloader as before: the input tuple has been expanded to load the `title`, `ingredients` and `instructions` as text along with the other fields already present

Files can be found here: https://drive.google.com/drive/folders/14brtR12WlZ8fqvRttcv43wXkOfusSVUo?usp=sharing