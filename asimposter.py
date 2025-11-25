import tkinter as tk
from tkinter import ttk
from openai import OpenAI
from dotenv import load_dotenv
import os
import random
import json
from tqdm import tqdm
import numpy as np
import math 
import scipy.stats as stats

# Load API key from .env file, create openAI client
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

PLAYER_ASSERTIONS = {}
WORD_PROBABILITIES = {}
DESIRED_GEN_COUNT = 50
TEMPERATURE = 0.1 # for softmax, more confident probabilties since cosine similarity differs so little
SECRET_WORD = ""
SECRET_EMBED = ""

# Code that does statistical inference
# We use cosine similarity with vector embeddings using openAI as an approximation for P(G = guess | S = secret word (one of the words in our prediction))
def do_statistical_inference(guess):
    guess_embed = client.embeddings.create(
        input=f"{guess}",
        model="text-embedding-3-small"
    )
    query_embedding = np.array(guess_embed.data[0].embedding)

    query_norm = np.linalg.norm(query_embedding)
    word_norms = np.linalg.norm(WORD_EMBEDDINGS, axis=1)
    # LEFT is 2D, RIGHT is 1D
    cos_similarities = np.matmul(WORD_EMBEDDINGS, query_embedding) / (query_norm * word_norms)
    
    print(cos_similarities)

    # scale with TEMPERATURE value
    scaled_cos = cos_similarities / TEMPERATURE

    # use softmax formula to get likelihoods
    likelihoods = np.exp(scaled_cos) / np.exp(scaled_cos).sum()
    print(likelihoods)

    priors = np.array(list(WORD_PROBABILITIES.values()))
    print(priors)
    print()

    posterior_beliefs = (likelihoods * priors) / (likelihoods @ priors)
    print(posterior_beliefs)

    for word, prob in zip(WORD_LIST, posterior_beliefs):
        WORD_PROBABILITIES[word] = prob
    stat_uncertainty = calculate_stat_uncertainty()
    refresh_stat_window(stat_tree, stat_uncertainty)
    print(WORD_PROBABILITIES)

def calculate_stat_uncertainty():
    uncertainty = 0
    for word in WORD_PROBABILITIES:
        uncertainty += math.log2(1 / WORD_PROBABILITIES[word]) * WORD_PROBABILITIES[word]
    print(f"Uncertainty: {uncertainty}")
    return uncertainty

def calculate_imp_uncertainty():
    uncertainty = 0
    for p in IMPOSTER_PROBABILTIES:
        uncertainty += math.log2(1 / IMPOSTER_PROBABILTIES[p]) * IMPOSTER_PROBABILTIES[p]
    print(f"Uncertainty: {uncertainty}")
    return uncertainty

def refresh_imp_window(tree, uncertainty):
    uncertainty_imp_text.config(text=f"Current Uncertainty: {uncertainty}")
    # Clear old rows
    for item in tree.get_children():
        tree.delete(item)

    print()
    sorted_players = sorted(
        list(IMPOSTER_PROBABILTIES.keys()),
        key=lambda p: IMPOSTER_PROBABILTIES[p],
        reverse=True
    )

    for rank, player_id in enumerate(sorted_players, start=1):
        prob = IMPOSTER_PROBABILTIES[player_id]
        tree.insert(
            "",
            "end",
            values=(rank, player_id, prob, PLAYER_ASSERTIONS[player_id])
        )

def refresh_stat_window(tree, uncertainty):
    uncertainty_stat_text.config(text=f"Current Uncertainty: {uncertainty}")

    # Clear old rows
    for item in tree.get_children():
        tree.delete(item)
    
     # Sort words by current probability (posterior) descending
    sorted_words = sorted(
        WORD_LIST,
        key=lambda word: WORD_PROBABILITIES[word],
        reverse=True
    )

    # Repopulate
    for idx, word in enumerate(sorted_words, start=1):
        prob = WORD_PROBABILITIES[word]
        tree.insert(
            "",
            "end",
            values=(idx, word, f"{prob:.4f}")  # format to 4 decimal places
        )

def do_imposter_inference(player_index, text):
   # Model as Gaussian distributions for simplicity

   # We expect non-imposters to have higher cosine similarity scores on average to the real secret word.
    non_imp_hint = stats.norm(0.4, 0.1)

   # We expect imposters to have lower cosine similarity scores on average, with greater deviations 
    imp_hint = stats.norm(0.3, 0.15)
    hint_embed = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    hint_vec = np.array(hint_embed.data[0].embedding)
   
    hint_norm = np.linalg.norm(hint_vec) 
    secret_norm = np.linalg.norm(SECRET_EMBED)
    cosine_similarity = float(np.dot(hint_vec, SECRET_EMBED) / (hint_norm * secret_norm))
    print(f"Cosine similarity of player {player_index}'s hint to secret:", cosine_similarity)

    unnormalized = {}
    for potential_imposter, prior_belief in IMPOSTER_PROBABILTIES.items():
        if potential_imposter == player_index:
            likelihood = imp_hint.pdf(cosine_similarity)
        else:
            likelihood = non_imp_hint.pdf(cosine_similarity)
        unnormalized[potential_imposter] = likelihood * prior_belief 

    total = sum(unnormalized.values())
    for j in IMPOSTER_PROBABILTIES.keys():
        IMPOSTER_PROBABILTIES[j] = unnormalized[j] / total

    print("Updated imposter probs:", IMPOSTER_PROBABILTIES)

    imp_uncertainty = calculate_imp_uncertainty()
    refresh_imp_window(imp_tree, imp_uncertainty)

    

def button_clicked(player_index, text_input, listbox):
    text = text_input.get()
    if text == "":
        return
    print(f"Button clicked for player {player_index}!")
    print(f"Entry received was {text}")
    PLAYER_ASSERTIONS[player_index].append(text)
    print(PLAYER_ASSERTIONS)
    listbox.insert(tk.END, f"{len(PLAYER_ASSERTIONS[player_index])}: {text}") 
    do_statistical_inference(text)
    do_imposter_inference(player_index, text)
    text_input.set("")

def build_ui(window, topic):
    window.title("Who's the imposter?")
    window.geometry("800x500")

    ## BEGINNING OF SCROLLABLE CONFIGURATION 
    container = tk.Frame(window)
    container.grid(row=0, column=0, sticky="nsew")

    # scrollbar
    scrollbar = tk.Scrollbar(container, orient="vertical")
    scrollbar.pack(side="right", fill="y")

    # canvas
    canvas = tk.Canvas(container, yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)

    scrollbar.config(command=canvas.yview)

    # frame INSIDE canvas
    scroll_frame = tk.Frame(canvas)
    scroll_frame.grid_columnconfigure(0, weight=1)
    scroll_window = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")

    def on_canvas_configure(event):
        canvas.itemconfig(scroll_window, width=event.width)

    canvas.bind("<Configure>", on_canvas_configure)

    # update canvas scroll region whenever frame size changes
    def update_scroll_region(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    scroll_frame.bind("<Configure>", update_scroll_region)

    # let window expand
    window.grid_rowconfigure(0, weight=1)
    window.grid_columnconfigure(0, weight=1)

    ## END OF SCROLLABLE CONFIGURATION 

    # Let the window expand horizontally as the geometry expands
    window.grid_columnconfigure(0, weight=1)
    # Minimum shrinking size
    window.minsize(400, 300)


    top_label = tk.Label(scroll_frame, text=f"The chosen category is {topic}", font=("Arial", 18, "bold"))
    top_label.grid(row=0, column=0, columnspan=3, sticky="ew", padx=10, pady=10)

    for i in range(1, NUM_PLAYERS + 1, 1):
        # Create frame (like a div element) surrounding each player frame
        player_frame = tk.Frame(scroll_frame, borderwidth=2, relief="raised", padx=5, pady=5)
        player_frame.grid(row=i, column=0, padx=10, pady=10, sticky="ew")
        player_frame.grid_columnconfigure(1, weight=1)
        # tk_vars which hold entry text and label text
        text_holder = tk.StringVar(value="Write here!")
        label_holder = tk.StringVar(value=f"Player {i}")
        # Set label for player
        label = tk.Label(player_frame, textvariable=label_holder)
        label.grid(row=0, column=0)

        # Set entry box
        entry = tk.Entry(player_frame, textvariable=text_holder)
        entry.grid(row=0, column=1, sticky="ew", padx=5) # stretch horizontally = sticky="ew"

        listbox = tk.Listbox(player_frame)
        listbox.grid(row=1, column=0, columnspan=3, padx=5, sticky="ew")

        # Set button
        button = tk.Button(player_frame, text="Click Me", command=lambda p_idx=i, t_h = text_holder, lb = listbox: button_clicked(p_idx, t_h, lb))
        button.grid(row=0, column=2, sticky="e", padx=5) # stick to east (right) = sticky = "e"


# Gets the player count from user!
def get_players():
    while True:
        response = input("How many players are participating? ")
        try:
            num = int(response)

            if num <= 0:
                print("Please enter a positive integer.")
                continue  # ask again

            for i in range(1, num + 1):
                PLAYER_ASSERTIONS[i] = []
            print(PLAYER_ASSERTIONS)
            return num  # valid → exit the loop and return value
        
        except ValueError:
            print("Please enter a valid integer.")
            # loop repeats automatically
    
def get_game_info():
    while True:
        response = input("Would you like to choose the topic? y/n: ")
        choice = response

        if choice != 'y' and choice != 'n':
            print("Please enter y/n.")
            continue  # ask again

        if choice == 'y':
            topic = input("Enter in your desired topic: ")
            print(f"Topic chosen: {topic}")

        if choice == 'n':
            random_topics = [
                'food',
                'drinks',
                'technology',
                'sports',
                'animals',
                'college',
                'occupations',
                'plants',
                'house items'
            ]
            topic = random.choice(random_topics)
        
        secret = input("What should be the secret word/phrase? ")
        return topic, secret
        
def prepare_statistical_analysis(topic, secret):
    generated_r = client.responses.create(
        model="gpt-5-mini",
        input=f"Given the topic '{topic}', generate {DESIRED_GEN_COUNT} common words (or short phrases of 2 or three words) that could possibly be the underlying secret phrase, no duplicates, and return them in JSON. It should be formatted as follows:\n{{\n  \"words\": [\"example1\", \"example2\"]\n}}"
    )
    text = generated_r.output[1].content[0].text
    data = json.loads(text)
    print(type(data))      # should be dict
    list_words = data["words"]
    print(list_words)   # list of words

    if secret not in list_words:
        list_words.pop()
        list_words.append(secret)
    """
    data = {
       'words': [
       'sofa', 'couch', 'chair', 'table', 'bed', 'mattress', 'pillow', 'blanket', 'quilt', 'duvet', 'sheet', 'rug', 'carpet', 'curtain', 'blinds', 'mirror', 'lamp', 'clock', 'picture', 'frame', 'shelf', 'bookshelf', 'cabinet', 'cupboard', 'drawer', 'wardrobe', 'dresser', 'nightstand', 'desk', 'television', 'fridge', 'refrigerator', 'stove', 'oven', 'microwave', 'toaster', 'blender', 'kettle', 'faucet', 'sink', 'dishwasher', 'pantry', 'toilet', 'bathtub', 'shower', 'towel', 'trashcan', 'broom', 'vacuum', 'plant'
       ]
    }
    list_words = data['words']
    """


    global WORD_PROBABILITIES
    WORD_PROBABILITIES = {word: 1 / len(list_words) for word in list_words}

    global IMPOSTER_PROBABILTIES
    IMPOSTER_PROBABILTIES = {i: 1 / NUM_PLAYERS for i in range(1, NUM_PLAYERS + 1)}

    print(WORD_PROBABILITIES)
    BATCH_SIZE = 256
    MODEL_NAME = "text-embedding-3-small"
    temp_embeddings_list = []
    for i in tqdm(range(0, len(list_words), BATCH_SIZE)):
        batch = list_words[i:i + BATCH_SIZE]
        embeds = client.embeddings.create(
            input=batch,
            model=MODEL_NAME,
        )
        for embed in embeds.data:
            temp_embeddings_list.append(embed.embedding)
    
    global SECRET_EMBED
    secret_embed = client.embeddings.create(
        input=secret,
        model=MODEL_NAME,
    )
    SECRET_EMBED = np.array(secret_embed.data[0].embedding)

    """
    for word in tqdm(list_words):
        WORD_PROBABILITIES[word] = 1 / len(list_words)
        embed = client.embeddings.create(
            input=f"{word}",
            model="text-embedding-3-small"
        )
        temp_embeddings_list.append(embed.data[0].embedding) 
    """
    print(temp_embeddings_list)

    global WORD_EMBEDDINGS
    WORD_EMBEDDINGS = np.array(temp_embeddings_list)
    print(WORD_EMBEDDINGS)
    print(WORD_PROBABILITIES)

    debug_similarity_distribution()
    return list_words

def debug_similarity_distribution():
    # Cosine similarities between candidate words and the secret
    norms_words = np.linalg.norm(WORD_EMBEDDINGS, axis=1)
    norm_secret = np.linalg.norm(SECRET_EMBED)
    sims = (WORD_EMBEDDINGS @ SECRET_EMBED) / (norms_words * norm_secret)

    print("Cosine similarities between SECRET and candidate words:")
    print("min:", float(sims.min()))
    print("max:", float(sims.max()))
    print("mean:", float(sims.mean()))
    print("std:", float(sims.std()))

# Buggy implementation
def build_statistical_window(window, topic):
    # Create second window
    stat_win = tk.Toplevel(window)
    stat_win.title(f"Word Likelihood Analysis – {topic}")
    stat_win.minsize(500, 300)

    top_label_text = f"Current Uncertainty: ~"
    global uncertainty_stat_text
    uncertainty_stat_text = tk.Label(stat_win, text=top_label_text)
    uncertainty_stat_text.grid(row=0, column=0, columnspan=2, pady=(10, 5))

    # Allow the Treeview to expand with the window
    stat_win.grid_rowconfigure(1, weight=1)
    stat_win.grid_columnconfigure(0, weight=1)

    # Define columns: Rank + Word + Probability
    columns = ("rank", "word", "secretword")
    tree = ttk.Treeview(stat_win, columns=columns, show="headings", height=15)

    # Headings
    tree.heading("rank", text="Rank")
    tree.heading("word", text="Word")
    tree.heading("secretword", text="P(secret word | data)")

    # Column widths / alignment
    tree.column("rank", width=50, anchor="center")
    tree.column("word", width=150, anchor="w")
    tree.column("secretword", width=150, anchor="e")

    # Add scrollbar
    scrollbar = ttk.Scrollbar(stat_win, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)

    # Layout
    tree.grid(row=1, column=0, sticky="nsew")
    scrollbar.grid(row=1, column=1, sticky="ns")

    # Populate with your WORD_LIST (for now: just rank in order, prob placeholder)
    for idx, word in enumerate(WORD_LIST, start=1):
        # third value is a placeholder; later you can plug in a real probability
        tree.insert("", "end", values=(idx, word, WORD_PROBABILITIES[word]))

    # Return the tree in case you want to update it later
    return tree

def build_imposter_prediction(window):
    imp_window = tk.Toplevel(window)
    imp_window.title("Determining the Imposter")
    imp_window.minsize(500, 400)

    top_label_text = f"Current Uncertainty: ~"
    global uncertainty_imp_text
    uncertainty_imp_text = tk.Label(imp_window, text=top_label_text)
    uncertainty_imp_text.grid(row=0, column=0, columnspan=2, pady=(10, 5))

    imp_window.grid_rowconfigure(1, weight=1)
    imp_window.grid_columnconfigure(0, weight=1)

    # Define columns: Rank + Word + Probability
    columns = ("rank", "player#", "imp_prob", "past_guesses")
    tree = ttk.Treeview(imp_window, columns=columns, show="headings", height=15)

    tree.heading("rank", text="Rank")
    tree.heading("player#", text="Player #")
    tree.heading("imp_prob", text="Imposter Probability")
    tree.heading("past_guesses", text="Past Guesses")

    tree.column("rank", width=50, anchor="center")
    tree.column("player#", width=50, anchor="center")
    tree.column("imp_prob",  width=100, anchor="e")
    tree.column("past_guesses", width=200, anchor="e")

    # Add scrollbar
    scrollbar = ttk.Scrollbar(imp_window, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)

    tree.grid(row=1, column=0, sticky="nsew")
    scrollbar.grid(row=1, column=1, sticky="ns")

    for i in range(NUM_PLAYERS):
        tree.insert("", "end", values=('~', i + 1, f"{IMPOSTER_PROBABILTIES[i + 1]:.4f}", ""))

    return tree


def main():

    # Get amount of players that are playing the game from the user 
    global NUM_PLAYERS
    NUM_PLAYERS = get_players()
    
    # Get the current game topic
    topic, secret = get_game_info()

    global SECRET_WORD
    SECRET_WORD = secret

    # Prepare list of possible words for imposter
    global WORD_LIST
    WORD_LIST = prepare_statistical_analysis(topic, secret)
    
    # Populate window and UI
    window = tk.Tk()
    build_ui(window, topic)
    global stat_tree
    stat_tree = build_statistical_window(window, topic)
    global imp_tree
    imp_tree = build_imposter_prediction(window)

    print(IMPOSTER_PROBABILTIES)

    window.mainloop()

if __name__ == "__main__":
    main()