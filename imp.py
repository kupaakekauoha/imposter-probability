import tkinter as tk
from tkinter import ttk

NUM_PLAYERS = 3  # change this if you want more/less players

# Globals so handlers can access them
PLAYER_VARS = []
PLAYER_HISTORY = []
PLAYER_LISTBOXES = []


def save_guess(player_index: int):
    """
    Save the current guess for the given player and show it in that player's listbox.
    """
    var = PLAYER_VARS[player_index]
    guess = var.get().strip()

    if not guess:
        return  # ignore empty guesses

    # Save in memory
    PLAYER_HISTORY[player_index].append(guess)

    # Add to that player's listbox
    lb = PLAYER_LISTBOXES[player_index]
    entry_number = len(PLAYER_HISTORY[player_index])
    display_text = f"{entry_number:02d}. {guess}"
    lb.insert(tk.END, display_text)

    # Clear the entry after saving
    var.set("")


def on_enter_pressed(event, player_index: int):
    """
    Called when Enter is pressed in a player's entry widget.
    """
    save_guess(player_index)
    return "break"  # prevents default "ding" or extra behavior


def build_ui(root):
    root.title("Player Guesses Tracker")
    root.geometry("600x500")

    # Let column 1 expand horizontally
    root.columnconfigure(1, weight=1)

    # Create rows for each player
    for i in range(NUM_PLAYERS):
        player_label_text = f"Player {i + 1} Guess:"

        # Label
        lbl = ttk.Label(root, text=player_label_text)
        lbl.grid(row=i * 2, column=0, padx=10, pady=4, sticky="e")

        # Entry (text box)
        entry = ttk.Entry(root, textvariable=PLAYER_VARS[i])
        entry.grid(row=i * 2, column=1, padx=10, pady=4, sticky="we")

        # Bind Enter key to save for this player
        entry.bind("<Return>", lambda event, idx=i: on_enter_pressed(event, idx))

        # Save button
        btn = ttk.Button(root, text="Save Guess",
                         command=lambda idx=i: save_guess(idx))
        btn.grid(row=i * 2, column=2, padx=10, pady=4, sticky="w")

        # Listbox for this player's history
        lb = tk.Listbox(root, height=4)
        lb.grid(row=i * 2 + 1, column=0, columnspan=3,
                padx=20, pady=(0, 8), sticky="we")

        PLAYER_LISTBOXES.append(lb)


def main():
    global PLAYER_VARS, PLAYER_HISTORY

    root = tk.Tk()

    # Create one StringVar and one history list per player
    PLAYER_VARS = [tk.StringVar() for _ in range(NUM_PLAYERS)]
    PLAYER_HISTORY = [[] for _ in range(NUM_PLAYERS)]

    build_ui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
