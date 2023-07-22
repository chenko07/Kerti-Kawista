import tkinter as tk
    
root = tk.Tk()
root.geometry("390x844")
root.title("Kerti Kawista")
root.overrideredirect(True)

frame = tk.Frame(root, width-390, height= 844, bg="#000")
frame.pack_propagate(False)
frame.pack()

root.mainloop()