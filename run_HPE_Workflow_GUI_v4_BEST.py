import tkinter as tk
from tkinter import ttk, messagebox
import subprocess

# ==============================
# Script Execution Function
# ==============================
def run_pipeline():
    dt = dt_var.get()
    numsteps = numsteps_var.get()
    #geom = geom_var.get()
    voltage = voltage_var.get()

    # --- Validate Inputs ---
    try:
        float(dt)
        int(numsteps)
        float(voltage)
        #int(geom)
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values for all parameters.")
        return

    # --- Build Script List Based on Checkboxes ---
    scripts = []
    if step_vars[0].get():
        scripts.append(["bash", "run_Conc3D_to_Conc1D.sh", str(dt), str(numsteps)])
    if step_vars[1].get():
        scripts.append(["bash", "run_FitPolynomial_x.sh", str(dt), str(numsteps)])
    if step_vars[2].get():
        scripts.append(["bash", "run_MAIN.sh", str(dt), str(numsteps), str(voltage)])
    
    if not scripts:
        messagebox.showwarning("No Steps Selected", "Please select at least one step to run.")
        return

    console_text.set("🚀 Starting selected workflow steps...\n")
    root.update()

    progress_bar["maximum"] = len(scripts)
    progress_bar["value"] = 0

    for i, script in enumerate(scripts, start=1):
        console_text.set(console_text.get() + f"\n▶ Step {i}/{len(scripts)}: {' '.join(script)}\n")
        root.update()
        try:
            result = subprocess.run(script, capture_output=True, text=True, check=True)
            console_text.set(console_text.get() + result.stdout + "\n")
        except subprocess.CalledProcessError as e:
            console_text.set(console_text.get() + f"❌ Error in {' '.join(script)}: {e.stderr}\n")
            messagebox.showerror("Execution Failed", f"Script {script[0]} failed.\nCheck console output.")
            return

        # Update progress bar
        progress_bar["value"] = i
        root.update_idletasks()

    console_text.set(console_text.get() + "\n✅ Selected steps completed successfully!\n")
    messagebox.showinfo("Done", "Selected computations completed. Check output folders for results.")


# ==============================
# GUI Layout & Theme
# ==============================
root = tk.Tk()
root.title("Hierarchical Porous Electrode – Workflow Runner")
root.geometry("780x650")
root.configure(bg="#0a2445")  # Dark background

# --- Title ---
title_label = tk.Label(root, text="⚡ Hierarchical Porous Electrode Simulation ⚡",
                       font=("Helvetica", 20, "bold"),
                       fg="#66fcf1", bg="#0b0c10")
title_label.pack(pady=10)

subtitle_label = tk.Label(root, text="Select parameters & steps, then run the workflow",
                          font=("Helvetica", 12), fg="#c5c6c7", bg="#0b0c10")
subtitle_label.pack()

# --- Input Fields ---
input_frame = tk.Frame(root, bg="#0b0c10")
input_frame.pack(pady=10)

labels_fg = "#45a29e"
tk.Label(input_frame, text="Time step (dt):", font=("Helvetica", 12, "bold"), fg=labels_fg, bg="#0b0c10").grid(row=0, column=0, padx=10, pady=5, sticky="e")
dt_var = tk.StringVar(value="1e-06")
tk.Entry(input_frame, textvariable=dt_var, width=15, bg="#1f2833", fg="white", insertbackground="white").grid(row=0, column=1)

tk.Label(input_frame, text="Number of steps:", font=("Helvetica", 12, "bold"), fg=labels_fg, bg="#0b0c10").grid(row=1, column=0, padx=10, pady=5, sticky="e")
numsteps_var = tk.StringVar(value="500")
tk.Entry(input_frame, textvariable=numsteps_var, width=15, bg="#1f2833", fg="white", insertbackground="white").grid(row=1, column=1)

#tk.Label(input_frame, text="Electrode Geometry No.:", font=("Helvetica", 12, "bold"), fg=labels_fg, bg="#0b0c10").grid(row=2, column=0, padx=10, pady=5, sticky="e")
#geom_var = tk.StringVar(value="1")
#tk.Entry(input_frame, textvariable=geom_var, width=15, bg="#1f2833", fg="white", insertbackground="white").grid(row=2, column=1)

tk.Label(input_frame, text="Applied Voltage (V):", font=("Helvetica", 12, "bold"), fg=labels_fg, bg="#0b0c10").grid(row=3, column=0, padx=10, pady=5, sticky="e")
voltage_var = tk.StringVar(value="1.5")
#tk.Entry(input_frame, textvariable=voltage_var, width=15, bg="#1f2833", fg="white", insertbackground="white").grid(row=3, column=1)
tk.Entry(input_frame, textvariable=voltage_var, width=15, bg="#1f2833", fg="white", insertbackground="white").grid(row=2, column=1)


# --- Checkboxes ---
steps = [
    "1. run_Conc3D_to_Conc1D.sh",
    "2. run_FitPolynomial_x.sh",
    "3. run_MAIN.sh",
    #"3. run_Interpolate_FittedPoly",
    #"4. plot_Conc4Geom.py",
    #"5. run_capacitance_vs_Geom.sh",
    #"6. run_capacity_vs_Geom.sh",
    #"7. run_capacitance_vs_Geom.sh",
    #"8. runEnergyPowerRateCharge.sh",
    #"9. runRagonePlot.sh"
]

steps_frame = tk.LabelFrame(root, text="✅ Select Workflow Steps", font=("Helvetica", 14, "bold"),
                            fg="#f85a3e", bg="#0b0c10", padx=10, pady=10)
steps_frame.pack(pady=10, fill="x")

step_vars = []
for step in steps:
    var = tk.BooleanVar(value=True)
    chk = tk.Checkbutton(steps_frame, text=step, variable=var, font=("Helvetica", 11),
                         fg="#66fcf1", bg="#0b0c10", selectcolor="#1f2833", activebackground="#1f2833")
    chk.pack(anchor="w")
    step_vars.append(var)

# --- Progress Bar ---
progress_bar = ttk.Progressbar(root, length=600, mode='determinate')
progress_bar.pack(pady=10)

# --- Run Button ---
run_button = tk.Button(root, text="▶ Run Selected Steps", font=("Helvetica", 14, "bold"),
                       bg="#f85a3e", fg="white", activebackground="#c0392b",
                       relief="raised", command=run_pipeline)
run_button.pack(pady=10)

# --- Console Output ---
console_text = tk.StringVar()
console_box = tk.Text(root, height=10, width=90, bg="#1f2833", fg="#f8f8f2", wrap="word")
console_box.pack(padx=10, pady=10)

def update_console(*args):
    console_box.delete(1.0, tk.END)
    console_box.insert(tk.END, console_text.get())
console_text.trace("w", update_console)

root.mainloop()

