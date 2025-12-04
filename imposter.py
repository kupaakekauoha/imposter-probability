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

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)
DESIRED_GEN_COUNT = 100
TEMPERATURE = 0.2

class GameState:
    def __init__(self):
        # int | None is a TYPE hint (could be int or None type)
        self.num_players: int | None = None
        self.topic: str | None = None
        self.secret_word: str | None = None
        self.player_assertions: dict[int, list[str]] = {}
        self.imposter_probabilities: dict[int, float] = {}
        self.word_probabilities: dict[str, float] = {}
        self.word_list: list[str] = []
        self.word_embeddings: np.ndarray | None = None
        self.secret_embed: np.ndarray | None = None

class UIState:
    def __init__(self):
        self.window = None
        self.stat_window = None
        self.stat_tree = None
        self.stat_uncertainty_label = None
        self.imp_window = None
        self.imp_tree = None
        self.imp_uncertainty_label = None

# Gets the player count from user!
def get_players():
    while True:
        response = input("How many players are participating? ")
        try:
            num = int(response)

            if num <= 0:
                print("Please enter a positive integer.")
                continue  # ask again

            if num > 50:
                print("Please enter a smaller amount of participants!")
                continue  # ask again

            return num  # valid → exit the loop and return value
        
        except ValueError:
            print("Please enter a valid integer.")
            # loop repeats automatically

def get_game_info():
    while True:
        choice = input("Would you like to choose the topic? y/n: ")

        if choice != 'y' and choice != 'n':
            print("Please enter y/n.")
            continue  # ask again

        if choice == 'y':
            topic = input("Enter in your desired topic: ")
        else:
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
        print(f"Topic chosen: {topic}")
        secret = input("What should be the secret word/phrase? ")
        return topic, secret

def prepare_statistical_analysis(game):
    generated_r = client.responses.create(
        model="gpt-5-mini",
        input=f"Given the topic '{game.topic}', generate {DESIRED_GEN_COUNT} common words (or short phrases of 2 or three words) that could possibly be the underlying secret phrase, no duplicates, and return them in JSON. It should be formatted as follows:\n{{\n  \"words\": [\"example1\", \"example2\"]\n}}"
    )
    # Constructs the list of candidate words to make predictions on 
    text = generated_r.output[1].content[0].text
    data = json.loads(text)
    print(type(data))      # should be dict
    word_list = data["words"]
    print(word_list)   # list of words
    if game.secret_word not in word_list:
        word_list.pop()
        word_list.append(game.secret_word)
    game.word_list = word_list
    ###
    game.word_probabilities = {word: 1 / len(game.word_list) for word in game.word_list}
    game.imposter_probabilities = {i: 1 / game.num_players for i in range(1, game.num_players + 1)}

    BATCH_SIZE = 256
    MODEL_NAME = "text-embedding-3-small"

    temp_embeddings_list = []
    for i in tqdm(range(0, len(game.word_list), BATCH_SIZE)):
        batch = game.word_list[i:i + BATCH_SIZE]
        embeds = client.embeddings.create(
            input=batch,
            model=MODEL_NAME,
        )
        for embed in embeds.data:
            temp_embeddings_list.append(embed.embedding)
    game.word_embeddings = np.array(temp_embeddings_list)

    secret_embed = client.embeddings.create(
        input=game.secret_word,
        model=MODEL_NAME,
    )
    game.secret_embed = np.array(secret_embed.data[0].embedding)
    print(game.word_embeddings)
    print(game.word_probabilities)
    return

def build_ui(ui_p, game):
    ui_p.window.title("Game: Who's The Imposter?")
    ui_p.window.geometry("800x500")

    ## BEGINNING OF SCROLLABLE CONFIGURATION 
    container = tk.Frame(ui_p.window)
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
    ui_p.window.grid_rowconfigure(0, weight=1)
    ui_p.window.grid_columnconfigure(0, weight=1)

    ## END OF SCROLLABLE CONFIGURATION 

    # Let the window expand horizontally as the geometry expands
    ui_p.window.grid_columnconfigure(0, weight=1)
    # Minimum shrinking size
    ui_p.window.minsize(400, 300)

    # Label on the top
    top_label = tk.Label(scroll_frame, text=f"The chosen category is {game.topic}", font=("Arial", 18, "bold"))
    top_label.grid(row=0, column=0, columnspan=3, sticky="ew", padx=10, pady=10)

    for i in range(1, game.num_players + 1, 1):
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
        button = tk.Button(player_frame, text="Click Me", command=lambda p_idx=i, t_h = text_holder, lb = listbox: button_clicked(p_idx, t_h, lb, game, ui_p))
        button.grid(row=0, column=2, sticky="e", padx=5) # stick to east (right) = sticky = "e"

def button_clicked(p_index, t_input, listbox, game, ui_p):
    text = t_input.get()
    if text == "":
        return
    print(f"Button clicked for player {p_index}!")
    print(f"Entry received was {text}")
    game.player_assertions[p_index].append(text)
    print(game.player_assertions)
    listbox.insert(tk.END, f"{len(game.player_assertions[p_index])}: {text}") 
    do_statistical_inference(text, game, ui_p)
    do_imposter_inference(p_index, text, game, ui_p)
    t_input.set("")

def do_imposter_inference(p_index, text, game, ui_p):
    # Model as Gaussian distributions for simplicity

   # We expect non-imposters to have higher cosine similarity scores on average to the real secret word.
    non_imp_hint = stats.norm(0.4, 0.15)

   # We expect imposters to have lower cosine similarity scores on average, with greater deviations 
    imp_hint = stats.norm(0.24, 0.1)
    hint_embed = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    hint_vec = np.array(hint_embed.data[0].embedding)
   
    hint_norm = np.linalg.norm(hint_vec) 
    secret_norm = np.linalg.norm(game.secret_embed)
    cosine_similarity = float(np.dot(hint_vec, game.secret_embed) / (hint_norm * secret_norm))
    print(f"Cosine similarity of player {p_index}'s hint to secret:", cosine_similarity)

    unnormalized = {}
    for potential_imposter, prior_belief in game.imposter_probabilities.items():
        if potential_imposter == p_index:
            likelihood = imp_hint.pdf(cosine_similarity)
        else:
            likelihood = non_imp_hint.pdf(cosine_similarity)
        unnormalized[potential_imposter] = likelihood * prior_belief 

    total = sum(unnormalized.values())
    for j in game.imposter_probabilities.keys():
        game.imposter_probabilities[j] = unnormalized[j] / total

    print("Updated imposter probs:", game.imposter_probabilities)

    imp_uncertainty = calculate_imp_uncertainty(game)
    refresh_imp_window(game, ui_p, imp_uncertainty)

def calculate_imp_uncertainty(game):
    uncertainty = 0
    for p in game.imposter_probabilities:
        uncertainty += math.log2(1 / game.imposter_probabilities[p]) * game.imposter_probabilities[p]
    print(f"Uncertainty: {uncertainty}")
    return uncertainty

def refresh_imp_window(game, ui_p, imp_uncertainty):
    ui_p.imp_uncertainty_label.config(text=f"Current Uncertainty: {imp_uncertainty}")
    # Clear old rows
    for item in ui_p.imp_tree.get_children():
        ui_p.imp_tree.delete(item)
    sorted_players = sorted(
        list(game.imposter_probabilities.keys()),
        key=lambda p: game.imposter_probabilities[p],
        reverse=True
    )
    for rank, player_id in enumerate(sorted_players, start=1):
        prob = game.imposter_probabilities[player_id]
        ui_p.imp_tree.insert(
            "",
            "end",
            values=(rank, player_id, prob, game.player_assertions[player_id])
        )

def do_statistical_inference(hint, game, ui_p):
    hint_embed = client.embeddings.create(
        input=f"{hint}",
        model="text-embedding-3-small"
    )
    query_embedding = np.array(hint_embed.data[0].embedding)

    query_norm = np.linalg.norm(query_embedding)
    word_norms = np.linalg.norm(game.word_embeddings, axis=1)
    # LEFT is 2D, RIGHT is 1D
    cos_similarities = np.matmul(game.word_embeddings, query_embedding) / (query_norm * word_norms)
    # scale with TEMPERATURE value
    scaled_cos = cos_similarities / TEMPERATURE
    # use softmax formula to get likelihoods
    likelihoods = np.exp(scaled_cos) / np.exp(scaled_cos).sum()

    priors = np.array(list(game.word_probabilities.values()))

    posterior_beliefs = (likelihoods * priors) / (likelihoods @ priors)
    for word, prob in zip(game.word_list, posterior_beliefs):
        game.word_probabilities[word] = prob
    stat_uncertainty = calculate_stat_uncertainty(game)
    refresh_stat_window(game, ui_p, stat_uncertainty)

def calculate_stat_uncertainty(game):
    uncertainty = 0
    for word in game.word_probabilities:
        uncertainty += math.log2(1 / game.word_probabilities[word]) * game.word_probabilities[word]
    print(f"Uncertainty: {uncertainty}")
    return uncertainty

def refresh_stat_window(game, ui_p, uncertainty):
    ui_p.stat_uncertainty_label.config(text=f"Current Uncertainty: {uncertainty}")
    # Clear old rows
    for item in ui_p.stat_tree.get_children():
        ui_p.stat_tree.delete(item)
    # Sort words by current probability (posterior) descending
    sorted_words = sorted(
        game.word_list,
        key=lambda word: game.word_probabilities[word],
        reverse=True
    )
    # Repopulate
    for idx, word in enumerate(sorted_words, start=1):
        prob = game.word_probabilities[word]
        ui_p.stat_tree.insert(
            "",
            "end",
            values=(idx, word, f"{prob:.4f}")  # format to 4 decimal places
        )

def build_statistical_window(ui_p, game):
    # Create second window
    ui_p.stat_window = tk.Toplevel(ui_p.window)
    ui_p.stat_window.title(f"Word Likelihood Analysis – {game.topic}")
    ui_p.stat_window.minsize(500, 300)

    top_label_text = f"Current Uncertainty: ~"
    ui_p.stat_uncertainty_label = tk.Label(ui_p.stat_window, text=top_label_text)
    ui_p.stat_uncertainty_label.grid(row=0, column=0, columnspan=2, pady=(10, 5))

    # Allow the Treeview to expand with the window
    ui_p.stat_window.grid_rowconfigure(1, weight=1)
    ui_p.stat_window.grid_columnconfigure(0, weight=1)

    # Define columns: Rank + Word + Probability
    columns = ("rank", "word", "secretword")
    tree = ttk.Treeview(ui_p.stat_window, columns=columns, show="headings", height=15)

    # Headings
    tree.heading("rank", text="Rank")
    tree.heading("word", text="Word")
    tree.heading("secretword", text="P(secret word | data)")

    # Column widths / alignment
    tree.column("rank", width=50, anchor="center")
    tree.column("word", width=150, anchor="w")
    tree.column("secretword", width=150, anchor="e")

    # Add scrollbar
    scrollbar = ttk.Scrollbar(ui_p.stat_window, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)

    # Layout
    tree.grid(row=1, column=0, sticky="nsew")
    scrollbar.grid(row=1, column=1, sticky="ns")

    # Populate with your WORD_LIST (for now: just rank in order, prob placeholder)
    for idx, word in enumerate(game.word_list, start=1):
        # third value is a placeholder; later you can plug in a real probability
        tree.insert("", "end", values=(idx, word, game.word_probabilities[word]))
    ui_p.stat_tree = tree

def build_imposter_window(ui_p, game):
    ui_p.imp_window = tk.Toplevel(ui_p.window)
    ui_p.imp_window.title("Determining the Imposter!")
    ui_p.imp_window.minsize(500, 400)

    top_label_text = f"Current Uncertainty: ~"
    ui_p.imp_uncertainty_label = tk.Label(ui_p.imp_window, text=top_label_text)
    ui_p.imp_uncertainty_label.grid(row=0, column=0, columnspan=2, pady=(10, 5))

    ui_p.imp_window.grid_rowconfigure(1, weight=1)
    ui_p.imp_window.grid_columnconfigure(0, weight=1)

    # Define columns: Rank + Word + Probability
    columns = ("rank", "player#", "imp_prob", "past_guesses")
    tree = ttk.Treeview(ui_p.imp_window, columns=columns, show="headings", height=15)

    tree.heading("rank", text="Rank")
    tree.heading("player#", text="Player #")
    tree.heading("imp_prob", text="Imposter Probability")
    tree.heading("past_guesses", text="Past Guesses")

    tree.column("rank", width=50, anchor="center")
    tree.column("player#", width=50, anchor="center")
    tree.column("imp_prob",  width=100, anchor="e")
    tree.column("past_guesses", width=200, anchor="e")

    # Add scrollbar
    scrollbar = ttk.Scrollbar(ui_p.imp_window, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)

    tree.grid(row=1, column=0, sticky="nsew")
    scrollbar.grid(row=1, column=1, sticky="ns")

    for i in range(game.num_players):
        tree.insert("", "end", values=('~', i + 1, f"{game.imposter_probabilities[i + 1]:.4f}", ""))
    
    ui_p.imp_tree = tree

def main():
    # Create game
    game = GameState()
    # Get number of players, the topic, and secret word.
    game.num_players = get_players()
    # Set player_assertions dictionary for later.
    for i in range(1, game.num_players + 1):
        game.player_assertions[i] = []

    game.topic, game.secret_word = get_game_info()
    prepare_statistical_analysis(game)

    ui_p = UIState()
    ui_p.window = tk.Tk()
    build_ui(ui_p, game)
    build_statistical_window(ui_p, game)
    build_imposter_window(ui_p, game)

    ui_p.window.mainloop()

if __name__ == "__main__":
    main()