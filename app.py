import streamlit as st
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data procession

def create_confusion_matrix(data):

    data = data["AggregatedEvaluationResults"]["ConfusionMatrix"]
    # Extract unique labels
    labels = sorted(set([item['GroundTruthLabel'] for item in data] + 
                        [item['PredictedLabel'] for item in data]))
    
    # Create a DataFrame with zeros
    df = pd.DataFrame(0, index=labels, columns=labels)
    
    # Fill the DataFrame with values
    for item in data:
        df.loc[item['GroundTruthLabel'], item['PredictedLabel']] = item['Value']
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(df, annot=True, fmt='.2f', cmap='Greens', ax=ax)
    
    # Set labels and title
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout and display the plot
    plt.tight_layout()
    st.pyplot(fig=fig)


# Streamlit
data = st.file_uploader("upload the json file with confusion matrix")

if data:
    data = json.load(data)
    create_confusion_matrix(data)




