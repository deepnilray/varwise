"""Tiny offline, Windows-friendly graphical interface for the real-data model."""
from pathlib import Path
import threading,webbrowser,datetime
import tkinter as tk
from tkinter import ttk,filedialog,messagebox
from engine import score,ROOT

root=tk.Tk();root.title('Infrared Discovery Engine — experimental research');root.geometry('650x335');root.minsize(570,300)
input_file=tk.StringVar(value=str(ROOT/'examples/known_training_lightcurves.csv'))
output_dir=tk.StringVar(value=str(ROOT/'results'/('run_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))))
status=tk.StringVar(value='Ready. Select a VIRAC2 DR3 Ks photometry CSV.')

frame=ttk.Frame(root,padding=18);frame.pack(fill='both',expand=True)
ttk.Label(frame,text='Infrared Discovery Engine',font=('Segoe UI',20,'bold')).pack(anchor='w')
ttk.Label(frame,text='Real-data machine learning · candidate ranking · not discovery confirmation',foreground='#755b29').pack(anchor='w',pady=(1,14))
for label,var,choose in [
    ('Measurements (CSV)',input_file,lambda:filedialog.askopenfilename(filetypes=[('CSV files','*.csv')])),
    ('Output folder',output_dir,lambda:filedialog.askdirectory())]:
    r=ttk.Frame(frame);r.pack(fill='x',pady=5)
    ttk.Label(r,text=label,width=19).pack(side='left')
    ttk.Entry(r,textvariable=var).pack(side='left',fill='x',expand=True,padx=8)
    def on_click(v=var,c=choose):
        selected=c()
        if selected:v.set(selected)
    ttk.Button(r,text='Browse',command=on_click).pack(side='left')

busy=ttk.Progressbar(frame,mode='indeterminate');busy.pack(fill='x',pady=(18,5))
button=ttk.Button(frame,text='Analyze real light curves');button.pack(anchor='w',pady=6)
ttk.Label(frame,textvariable=status,wraplength=595).pack(anchor='w',pady=(8,0))

def run():
    path=Path(input_file.get());out=Path(output_dir.get())
    if not path.is_file():messagebox.showerror('Missing CSV','Choose an existing CSV file');return
    button['state']='disabled';busy.start(12);status.set('Analyzing photometry and ranking candidates...')
    def worker():
        try:r=score(path,out,top=25)
        except Exception as ex:
            root.after(0,lambda ex=ex:(busy.stop(),button.config(state='normal'),status.set('Analysis failed.'),messagebox.showerror('Error',str(ex))))
            return
        def done():
            busy.stop();button.config(state='normal')
            status.set(f"Scored {r['scored_sources']}/{r['sources']} sources; {r['review_leads']} experimental leads. Opening report...")
            webbrowser.open((out/'candidate_report.html').resolve().as_uri())
        root.after(0,done)
    threading.Thread(target=worker,daemon=True).start()
button.config(command=run)
root.mainloop()
