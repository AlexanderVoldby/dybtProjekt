# train_and_evaluate.py

import os
from train_on_new_data import train_model
from load_model_and_predict_on_new_dataset import evaluate_model


def main():
    # Define explicit combinations of datasets and variants for training
    train_datasets = [     
        [("cars", "sd2.1"), ("cars", "real-fewshot")],
        [("pets", "sd2.1"), ("pets", "real-fewshot")],
        [("cars", "sd2.1"), ("pets", "sd2.1"), ("cars", "real-fewshot"), ("pets", "real-fewshot")]
    ]

    # Define evaluation datasets
    eval_dataset_combinations = [
        [("cars", "dd-fewshot"), ("cars", "real-fewshot")],
        [("cars", "sd2.1"), ("cars", "real-fewshot")],
        [("pets", "sd2.1"), ("pets", "real-fewshot")],
        [("pets", "dd-fewshot"), ("pets", "real-fewshot")]
    ]

    results = {}

    # Iterate over explicitly defined training combinations
    for train_combination in train_datasets:
        # Create a unique name for this combination
        combined_name = "__".join([f"{dataset}_{variant}" for dataset, variant in train_combination])
        model_name = f"combined_{combined_name}"

        # Check if the model already exists
        if os.path.exists(f"models/{model_name}.pth"):
            print(f"Model {model_name} already exists. Skipping training.")

        else:
            # Extract datasets and variants for training
            datasets = [dataset for dataset, variant in train_combination]
            variants = [variant for dataset, variant in train_combination]
            # Train the model on the current combination
            train_model(datasets, variants, model_name)

        # Evaluate the model on all specified evaluation datasets
        for eval_combination in eval_dataset_combinations:
            # Create a unique name for this combination
            eval_combined_name = "__".join([f"{dataset}_{variant}" for dataset, variant in eval_combination])

            # Extract datasets and variants for evaluation
            eval_datasets = [dataset for dataset, variant in eval_combination]
            eval_variants = [variant for dataset, variant in eval_combination]

            # Evaluate the model on the current combination
            metrics = evaluate_model(model_name, eval_datasets, eval_variants)

            # Store the results
            results[(model_name, eval_combined_name)] = metrics

    # Save the results to a CSV file
    results_path = "results.csv"
    with open(results_path, "w") as f:
        # Write the header
        f.write("Trained_On,Evaluated_On,Accuracy,Class,Precision,Recall,F1-Score\n")
        
        # Write metrics for each trained-on and evaluated-on combination
        for (trained_on, evaluated_on), metrics in results.items():
            # Overall accuracy
            accuracy = metrics["accuracy"]
            
            # Metrics for Class 0
            class_0_prec = metrics["class_0"]["precision"]
            class_0_rec = metrics["class_0"]["recall"]
            class_0_f1 = metrics["class_0"]["f1"]

            # Metrics for Class 1
            class_1_prec = metrics["class_1"]["precision"]
            class_1_rec = metrics["class_1"]["recall"]
            class_1_f1 = metrics["class_1"]["f1"]

            # Write metrics for Class 0
            f.write(f"{trained_on},{evaluated_on},{accuracy:.4f},0,{class_0_prec:.4f},{class_0_rec:.4f},{class_0_f1:.4f}\n")
            
            # Write metrics for Class 1
            f.write(f"{trained_on},{evaluated_on},{accuracy:.4f},1,{class_1_prec:.4f},{class_1_rec:.4f},{class_1_f1:.4f}\n")

    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
