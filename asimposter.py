import tkinter as tk
from tkinter import ttk
from openai import OpenAI
from dotenv import load_dotenv
import argparse
import os

def button_clicked(player_index, text):
    print(f"Button clicked for player {player_index}!")
    print(f"Entry received was {text.get()}")
    print()

def build_ui(window):
    window.title("Who's the imposter?")
    window.geometry("800x500")
    # Let the window expand horizontally as the geometry expands
    window.grid_columnconfigure(0, weight=1)
    # Minimum shrinking size
    window.minsize(400, 300)

    for i in range(1, num_players + 1, 1):
        # Create frame (like a div element) surrounding each player frame
        player_frame = tk.Frame(window, borderwidth=2, relief="raised", padx=5, pady=5)
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

        # Set button
        button = tk.Button(player_frame, text="Click Me", command=lambda p_idx=i, t_h = text_holder: button_clicked(p_idx, t_h))
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

            return num  # valid → exit the loop and return value
        
        except ValueError:
            print("Please enter a valid integer.")
            # loop repeats automatically


def main():
    # Get amount of players that are playing the game from the user 
    global num_players
    num_players = get_players()
    # Load API key from .env file
    load_dotenv()
    global API_KEY
    API_KEY = os.getenv("OPENAI_API_KEY")
    window = tk.Tk()
    build_ui(window)
    window.mainloop()



if __name__ == "__main__":
    main()
