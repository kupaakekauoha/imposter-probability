import tkinter as tk
from tkinter import ttk
from openai import OpenAI
from dotenv import load_dotenv
import os
import random
import json

PLAYER_ASSERTIONS = {}
WORD_PROBABILITIES = {}
DESIRED_GEN_COUNT = 50

# Code that does statistical inference

def button_clicked(player_index, text_input, listbox):
    text = text_input.get()
    if text == "":
        return
    print(f"Button clicked for player {player_index}!")
    print(f"Entry received was {text_input.get()}")
    PLAYER_ASSERTIONS[player_index].append(text)
    print(PLAYER_ASSERTIONS)
    listbox.insert(tk.END, f"{len(PLAYER_ASSERTIONS[player_index])}: {text}") 
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
            print(topic)
            return topic

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
            return topic
        
def prepare_statistical_analysis(topic):
    
    generated_r = client.responses.create(
        model="gpt-5-mini",
        input=f"Given the topic '{topic}', generate {DESIRED_GEN_COUNT} common words that could possibly be the underlying secret word, no duplicates, and return them in JSON. It should be formatted as follows:\n{{\n  \"words\": [\"example1\", \"example2\"]\n}}"
    )
    text = generated_r.output[1].content[0].text
    data = json.loads(text)
    print(type(data))      # should be dict
    list_words = data["words"]
    print(list_words)   # list of words
    """
    data = {
       'words': [
           'sofa', 'couch', 'chair', 'table', 'bed', 'mattress', 'pillow', 'blanket', 'quilt', 'duvet', 'sheet', 'rug', 'carpet', 'curtain', 'blinds', 'mirror', 'lamp', 'clock', 'picture', 'frame', 'shelf', 'bookshelf', 'cabinet', 'cupboard', 'drawer', 'wardrobe', 'dresser', 'nightstand', 'desk', 'television', 'fridge', 'refrigerator', 'stove', 'oven', 'microwave', 'toaster', 'blender', 'kettle', 'faucet', 'sink', 'dishwasher', 'pantry', 'toilet', 'bathtub', 'shower', 'towel', 'trashcan', 'broom', 'vacuum', 'plant'
        ]
    }
    """
    list_words = data["words"]

    for word in list_words:
        WORD_PROBABILITIES[word] = 1 / len(list_words)
    print(WORD_PROBABILITIES)
    return list_words

# Buggy implementation
def build_statistical_window(window, topic):
    # Create second window
    stat_win = tk.Toplevel(window)
    stat_win.title(f"Word Likelihood Analysis – {topic}")
    stat_win.minsize(400, 300)

    # Allow the Treeview to expand with the window
    stat_win.grid_rowconfigure(0, weight=1)
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
    tree.grid(row=0, column=0, sticky="nsew")
    scrollbar.grid(row=0, column=1, sticky="ns")

    # Populate with your WORD_LIST (for now: just rank in order, prob placeholder)
    for idx, word in enumerate(WORD_LIST, start=1):
        # third value is a placeholder; later you can plug in a real probability
        tree.insert("", "end", values=(idx, word, WORD_PROBABILITIES[word]))

    # Return the tree in case you want to update it later
    return tree

def main():
    # Load API key from .env file, create openAI client
    load_dotenv()
    global API_KEY
    API_KEY = os.getenv("OPENAI_API_KEY")
    global client
    client = OpenAI(api_key=API_KEY)

    # Get amount of players that are playing the game from the user 
    global NUM_PLAYERS
    NUM_PLAYERS = get_players()
    
    # Get the current game topic
    topic = get_game_info()

    # Prepare list of possible words for imposter
    global WORD_LIST
    WORD_LIST = prepare_statistical_analysis(topic)
    
    # Populate window and UI
    window = tk.Tk()
    build_ui(window, topic)
    build_statistical_window(window, topic)

    window.mainloop()

if __name__ == "__main__":
    main()