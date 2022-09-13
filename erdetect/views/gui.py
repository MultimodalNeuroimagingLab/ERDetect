#!/usr/bin/env python3
"""
Evoked response detection - GUI entry-point
=====================================================
GUI entry-point python script for the automatic detection of evoked responses in CCEP data.


Copyright 2022, Max van den Boom (Multimodal Neuroimaging Lab, Mayo Clinic, Rochester MN)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""



def open_gui():
    """

    """

    # Python might not be configured for tk, so by importing it only here, the
    # rest (functions and command-line wrapper) can run without trouble
    import tkinter as tk
    from tkinter import filedialog

    # defaults
    window_height = 500
    window_width = 640


    # open window
    win = tk.Tk()
    win.title('Evoked Response detection')
    win.geometry("{}x{}+{}+{}".format(window_width, window_height,
                                      int((win.winfo_screenwidth() / 2) - (window_width / 2)),
                                      int((win.winfo_screenheight() / 2) - (window_height / 2))))
    win.resizable(False, False)

    # window variables
    input_browse = tk.StringVar()

    # callbacks
    def input_browse_callback():
        folder_selected = filedialog.askdirectory(title='Open BIDS root directory', initialdir='~')
        if folder_selected is not None and folder_selected != '':
            input_browse.set(folder_selected)

    # elements
    lbl_input_browse = tk.Label(win, text="BIDS input directory:")
    lbl_input_browse.place(x=20, y=15)
    txt_input_browse = tk.Entry(win, textvariable=input_browse, width=60)
    txt_input_browse.place(x=20, y=40)
    btn_input_browse = tk.Button(win, text="Browse", command=input_browse_callback)
    btn_input_browse.place(x=20, y=70)


    # open window
    win.mainloop()
    exit()

