import customtkinter

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

root = customtkinter.CTk()
root.geometry("800x800")

def login():
    print("Login")

frame = customtkinter.CTkFrame(master=root, width=800, height=800)
frame.pack(pady=20, padx=60, fill="both", expand=True)

root.mainloop()