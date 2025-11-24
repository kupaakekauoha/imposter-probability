import tkinter as tk
from tkinter import ttk
from openai import OpenAI
from dotenv import load_dotenv
import os

def button_clicked(player_index, text_input, listbox):
    print(f"Button clicked for player {player_index}!")
    print(f"Entry received was {text_input.get()}")
    listbox.insert(tk.END, f"{text_input.get()}")
    

def build_ui(window):
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

    top_label = tk.Label(scroll_frame, text="The chosen category is ___", font=("Arial", 18, "bold"))
    top_label.grid(row=0, column=0, columnspan=3, sticky="ew", padx=10, pady=10)

    for i in range(1, num_players + 1, 1):
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
