import spacy
import pickle
import random
from spacy.training import Example  # Import Example for formatting training data

# Load the training data from a pickle file
train_data = pickle.load(open('train_data.pkl', 'rb'))
#print(train_data[0])  # Display the first training example for debugging

# Initialize a blank English NLP model
nlp = spacy.blank('en')

def train_model(train_data):
    # Check if 'ner' component is not in the pipeline and add it
    if 'ner' not in nlp.pipe_names:
        ner = nlp.add_pipe('ner', last=True)  # Add the NER component to the pipeline
    
    # Add entity labels to the NER model from the training data
    for _, annotations in train_data:
        for ent in annotations['entities']:
            ner.add_label(ent[2])  # Add the entity label

    # Disable other pipes to only train NER
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # Only train NER
        optimizer = nlp.begin_training()  # Initialize the optimizer
        for itn in range(10):  # Loop for the number of iterations
            print(f"Starting iteration {itn}")
            random.shuffle(train_data)  # Shuffle the training data
            losses = {}  # Initialize losses dictionary
            
            # Update the model with the training data
            for text, annotations in train_data:
                try:
                    # Create Example objects for training
                    example = Example.from_dict(nlp.make_doc(text), annotations)
                    nlp.update([example], drop=0.2, sgd=optimizer, losses=losses)  # Update with Example
                except Exception as e:
                    print(f"Error updating model: {e}")  # Handle errors
            
            print(losses)  # Print the losses for each iteration

# Call the training function
train_model(train_data)
nlp.to_disk('nlp_model')

#nlp_model = spacy.load('nlp_model')
#train_data[0][0]
#doc = nlp_model(train_data[0][0])
#for ent in doc.ents:
#    print(f'{ent.label_.upper():{30}}- {ent.text}')