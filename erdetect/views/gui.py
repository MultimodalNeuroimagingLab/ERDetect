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
import os
import logging
import threading

from ieegprep import VALID_FORMAT_EXTENSIONS
from ieegprep.bids import list_bids_datasets
from erdetect.core.config import load_config, get as cfg, set as cfg_set, rem as cfg_rem, create_default_config
from erdetect._erdetect import process_subset

def open_gui():
    """

    """

    # Python might not be configured for tk, so by importing it here, the
    # rest (functions and command-line wrapper) can run without trouble
    import tkinter as tk
    import tkinter.ttk as ttk
    from tkinter import filedialog

    #
    # the pre-process configuration dialog
    #
    class PreprocessDialog(object):

        reref_values_text = {'CAR': 'Common Average Re-refencing (CAR)', 'Henk': 'Henk'}
        reref_text_values = {v: k for k, v in reref_values_text.items()}

        def _update_early_refef_controls(self):
            new_state = 'normal' if self.early_reref.get() else 'disabled'
            self.lbl_early_reref_method.configure(state=new_state)
            self.cmb_early_reref_method.configure(state=new_state)
            self.lbl_early_reref_epoch.configure(state=new_state)

        def _update_combo_losefocus(self, event):
            self.root.focus()

        def __init__(self, parent):
            pd_window_height = 300
            pd_window_width = 550

            self.highpass = tk.IntVar(value=cfg('preprocess', 'high_pass'))
            self.early_reref = tk.IntVar(value=cfg('preprocess', 'early_re_referencing', 'enabled'))
            self.early_reref_method = tk.StringVar(value=self.reref_values_text[str(cfg('preprocess', 'early_re_referencing', 'method'))])
            self.early_reref_epoch_start = tk.DoubleVar(value=cfg('preprocess', 'early_re_referencing', 'stim_excl_epoch')[0])
            self.early_reref_epoch_end = tk.DoubleVar(value=cfg('preprocess', 'early_re_referencing', 'stim_excl_epoch')[1])

            #
            self.root = tk.Toplevel(parent)
            self.root.title('Preprocessing')
            self.root.geometry("{}x{}+{}+{}".format(pd_window_width, pd_window_height,
                                      int((win.winfo_screenwidth() / 2) - (pd_window_width / 2)),
                                      int((win.winfo_screenheight() / 2) - (pd_window_height / 2))))
            self.root.resizable(False, False)
            blank_icon = tk.PhotoImage(height=16, width=16)
            blank_icon.blank()
            self.root.iconphoto(False, blank_icon)

            #
            pd_y_pos = 10
            self.chk_highpass = tk.Checkbutton(self.root, text='High pass filtering (0.50Hz)', anchor="w", variable=self.highpass, onvalue=1, offvalue=0)
            self.chk_highpass.place(x=10, y=pd_y_pos, width=pd_window_width, height=30)
            pd_y_pos += 30
            self.chk_early_reref = tk.Checkbutton(self.root, text='Early re-referencing:', anchor="w", variable=self.early_reref, onvalue=1, offvalue=0, command=self._update_early_refef_controls)
            self.chk_early_reref.place(x=10, y=pd_y_pos, width=pd_window_width, height=30)
            pd_y_pos += 32
            early_reref_state = 'normal' if self.early_reref.get() else 'disabled'
            self.lbl_early_reref_method = tk.Label(self.root, text="Method", anchor='e', state=early_reref_state)
            self.lbl_early_reref_method.place(x=5, y=pd_y_pos + 2, width=205, height=20)
            self.cmb_early_reref_method = ttk.Combobox(self.root, textvariable=self.early_reref_method, values=list(self.reref_text_values.keys()), state=early_reref_state)
            self.cmb_early_reref_method.bind("<Key>", lambda e: "break")
            self.cmb_early_reref_method.bind("<<ComboboxSelected>>", self._update_combo_losefocus)
            self.cmb_early_reref_method.bind("<FocusIn>", self._update_combo_losefocus)
            self.cmb_early_reref_method.place(x=220, y=pd_y_pos, width=300, height=25)
            pd_y_pos += 32
            self.lbl_early_reref_epoch = tk.Label(self.root, text="Stim exclusion window", anchor='e', state=early_reref_state)
            self.lbl_early_reref_epoch.place(x=5, y=pd_y_pos + 2, width=205, height=20)
            self.txt_early_reref_epoch_start = ttk.Entry(self.root, textvariable=self.early_reref_epoch_start, state=early_reref_state, justify='center')
            self.txt_early_reref_epoch_start.place(x=220, y=pd_y_pos, width=70, height=25)
            self.lbl_early_reref_epoch_range = tk.Label(self.root, text="-", state=early_reref_state)
            self.lbl_early_reref_epoch_range.place(x=295, y=pd_y_pos, width=30, height=25)
            self.txt_early_reref_epoch_end = ttk.Entry(self.root, textvariable=self.early_reref_epoch_end, state=early_reref_state, justify='center')
            self.txt_early_reref_epoch_end.place(x=330, y=pd_y_pos, width=70, height=25)
            pd_y_pos += 32

            #
            tk.Button(self.root, text="OK", command=self.ok).place(x=10, y=pd_window_height - 40, width=120, height=30)
            tk.Button(self.root, text="Defaults", command=self.defaults).place(x=(pd_window_width - 100) / 2, y=pd_window_height - 35, width=100, height=25)
            tk.Button(self.root, text="Cancel", command=self.cancel).place(x=pd_window_width - 130, y=pd_window_height - 40, width=120, height=30)

            # modal window
            self.root.wait_visibility()
            self.root.grab_set()
            self.root.transient(parent)
            self.parent = parent
            self.root.focus()

        def ok(self):

            # update config
            cfg_set(self.highpass.get(), 'preprocess', 'high_pass')
            cfg_set(self.early_reref.get(), 'preprocess', 'early_re_referencing', 'enabled')
            cfg_set(self.reref_text_values[self.early_reref_method.get()], 'preprocess', 'early_re_referencing', 'method')

            #
            self.root.grab_release()
            self.root.destroy()

        def cancel(self):
            self.root.grab_release()
            self.root.destroy()

        def defaults(self):

            config_defaults = create_default_config()
            self.highpass.set(config_defaults['preprocess']['high_pass'])
            self.early_reref.set(config_defaults['preprocess']['early_re_referencing']['enabled'])
            self.early_reref_method.set(self.reref_values_text[config_defaults['preprocess']['early_re_referencing']['method']])
            self._update_early_refef_controls()



    #
    # the main window
    #

    # defaults
    window_width = 640
    window_height = 800
    cfg_set(1, 'preprocess', 'high_pass')
    cfg_set(1, 'preprocess', 'early_re_referencing', 'enabled')
    cfg_set('CAR', 'preprocess', 'early_re_referencing', 'method')

    # open window
    win = tk.Tk()
    #win.iconbitmap("myIcon.ico")
    win.title('Evoked Response detection')
    win.geometry("{}x{}+{}+{}".format(window_width, window_height,
                                      int((win.winfo_screenwidth() / 2) - (window_width / 2)),
                                      int((win.winfo_screenheight() / 2) - (window_height / 2))))
    win.resizable(False, False)

    # window variables
    datasets = None
    datasets_filtered_keys = None
    input_directory = tk.StringVar()
    subset_items = tk.StringVar()
    subset_filter = tk.StringVar()
    apply_bids_validator = tk.IntVar()
    processing_thread = None
    processing_thread_lock = threading.Lock()

    # callbacks
    def btn_input_browse_onclick():
        nonlocal datasets, datasets_filtered_keys

        initial_dir = input_directory.get()
        if not initial_dir:
            initial_dir = os.path.abspath(os.path.expanduser(os.path.expandvars('~')))

        folder_selected = filedialog.askdirectory(title='Open BIDS root directory', initialdir=initial_dir)
        if folder_selected is not None and folder_selected != '':
            input_directory.set(os.path.abspath(os.path.expanduser(os.path.expandvars(folder_selected))))
            datasets = None

            # reset search filters
            datasets_filtered_keys = None
            subset_filter.set('')

            # search for datasets
            try:
                datasets_found = list_bids_datasets(  folder_selected, VALID_FORMAT_EXTENSIONS,
                                                    strict_search=False,
                                                    only_subjects_with_subsets=True)

                # place the dataset list in a long format structure
                if len(datasets_found) > 0:
                    datasets = dict()
                    for subject in sorted(datasets_found.keys()):
                        for dataset in sorted(datasets_found[subject]):
                            datasets[dataset] = dict()

                            #
                            short_label = os.path.splitext(os.path.basename(os.path.normpath(dataset)))[0]
                            if short_label.endswith('_ieeg'):
                                short_label = short_label[0:-5]
                            if short_label.endswith('_eeg'):
                                short_label = short_label[0:-4]

                            #
                            datasets[dataset]['label'] = short_label
                            datasets[dataset]['selected'] = 0

            except (NotADirectoryError, ValueError):
                pass

            # if no datasets, set text and disable list
            if not datasets:
                subset_items.set(value=('  - Could not find datasets in the selected BIDS directory -',))
                lst_subsets.configure(background='systemWindowBackgroundColor', state='disabled')
                btn_subsets_all.config(state='disabled')
                btn_subsets_none.config(state='disabled')
                lbl_subsets_filter.config(state='disabled')
                txt_subsets_filter.config(state='disabled')
                btn_process.config(state='disabled', text='Process')

            # initially no selection, so disable process button
            btn_process.config(state='disabled', text='Start')

            #
            update_subset_list('')


    def update_subset_list(filter):
        nonlocal datasets, datasets_filtered_keys
        if datasets:

            if filter:
                filter = filter.lower()
                datasets_filtered_keys = [key for key, val in datasets.items() if filter in val['label'].lower()]
            else:
                datasets_filtered_keys = datasets.keys()

            # compile a list of labels to display
            lst_values = []
            for key in datasets_filtered_keys:
                lst_values.append(' ' + datasets[key]['label'])
            subset_items.set(value=lst_values)

            # enable controls
            lst_subsets.configure(background='systemTextBackgroundColor', state='normal')
            btn_subsets_all.config(state='normal')
            btn_subsets_none.config(state='normal')
            lbl_subsets_filter.config(state='normal')
            txt_subsets_filter.config(state='normal')

            # set the selections in the list
            lst_subsets.select_clear(0, tk.END)
            for index, key in enumerate(datasets_filtered_keys):
                if datasets[key]['selected']:
                    lst_subsets.selection_set(index)

    def lst_subsets_onselect(evt):
        nonlocal datasets, datasets_filtered_keys

        # update the dataset selection flags
        selected_indices = evt.widget.curselection()
        for index, key in enumerate(datasets_filtered_keys):
            datasets[key]['selected'] = index in selected_indices
        update_process_btn()

    def btn_subsets_all_onclick():
        for index, key in enumerate(datasets_filtered_keys):
            datasets[key]['selected'] = 1
        lst_subsets.selection_set(0, tk.END)
        update_process_btn()

    def btn_subsets_none_onclick():
        for index, key in enumerate(datasets_filtered_keys):
            datasets[key]['selected'] = 0
        lst_subsets.select_clear(0, tk.END)
        update_process_btn()

    def update_process_btn():
        nonlocal datasets
        datasets_to_analyze = [key for key, val in datasets.items() if val['selected']]

        if len(datasets_to_analyze) > 0:
            btn_process.config(state='normal')
            if len(datasets_to_analyze) == 1:
                btn_process.config(text='Start (1 set)')
            else:
                btn_process.config(text='Start (' + str(len(datasets_to_analyze)) + ' sets)')
        else:
            btn_process.config(state='disabled', text='Start')

    def txt_subsets_filter_onkeyrelease(evt):
        update_subset_list(subset_filter.get())

    def config_preprocessing_callback():
        dialog = PreprocessDialog(win)
        win.wait_window(dialog.root)

    def btn_process_start_onclick():
        nonlocal processing_thread, processing_thread_lock
        #txt_console.configure(background='systemTextBackgroundColor', state='normal')

        #
        datasets_to_analyze = [(val['label'], key) for key, val in datasets.items() if val['selected']]

        # create a thread to process the datasets
        processing_thread_lock.acquire()
        if processing_thread is None:
            processing_thread = threading.Thread(target=process_thread, args=(datasets_to_analyze,), daemon=False)
            processing_thread.start()
        else:
            print('Already started')
        processing_thread_lock.release()

        # disable controls
        btn_process.config(state='disabled')

        # TODO: show only sets to process and disable list
        lst_subsets.configure(background='systemWindowBackgroundColor', state='disabled')
        btn_subsets_all.config(state='disabled')
        btn_subsets_none.config(state='disabled')
        lbl_subsets_filter.config(state='disabled')
        txt_subsets_filter.config(state='disabled')
        # TODO: disable configuration buttons

    def process_thread(process_datasets):
        nonlocal processing_thread, processing_thread_lock

        # display subject/subset information
        txt_console.insert(tk.END, 'Participant(s) and subset(s) to process:\n')
        for (name, path) in process_datasets:
            txt_console.insert(tk.END, '    - ' + name + '\n')
        txt_console.insert(tk.END, '\n')

        # TODO: output directory
        output_dir = '~/Documents/ccepAgeOutput'
        output_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(output_dir)))

        # process
        for (val, path) in process_datasets:

            txt_console.insert(tk.END, '------ Processing ' + name + ' -------\n')
            logging.info('\n')

            # process
            path = os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
            try:
                #process_subset(path, output_dir, True)
                pass
            except RuntimeError:
                txt_console.insert(tk.END, 'Error while processing dataset, stopping...')
                # TODO: handle when error

        # empty space and end message
        txt_console.insert(tk.END, '\n\n')
        txt_console.insert(tk.END, '-----------      Finished running      -----------')

    #
        print(len(datasets))

        processing_thread_lock.acquire()
        processing_thread = None
        processing_thread_lock.release()


    # elements
    y_pos = 10
    tk.Label(win, text="BIDS input directory:", anchor='w').place(x=5, y=y_pos, width=window_width - 10, height=20)
    y_pos += 20 + 5
    txt_input_browse = tk.Entry(win, textvariable=input_directory, state='readonly')
    txt_input_browse.place(x=10, y=y_pos, width=window_width - 120, height=25)
    tk.Button(win, text="Browse...", command=btn_input_browse_onclick).place(x=window_width - 105, y=y_pos, width=95, height=25)

    y_pos += 37 + 5
    tk.Label(win, text="Subjects/subsets (click/highlight to include for processing):", anchor='w').place(x=5, y=y_pos, width=window_width - 10, height=20)
    y_pos += 20 + 5
    lst_subsets = tk.Listbox(win, listvariable=subset_items, selectmode="multiple", exportselection=False, state='disabled', background='systemWindowBackgroundColor')
    lst_subsets.place(x=10, y=y_pos, width=window_width - 40, height=200)
    lst_subsets.bind('<<ListboxSelect>>', lst_subsets_onselect)
    scr_subsets = tk.Scrollbar(win, orient='vertical', command=lst_subsets.yview)
    scr_subsets.place(x=window_width - 30, y=y_pos, width=20, height=200)
    lst_subsets['yscrollcommand'] = scr_subsets.set
    y_pos += 200 + 1
    btn_subsets_all = tk.Button(win, text="All", command=btn_subsets_all_onclick, state='disabled')
    btn_subsets_all.place(x=10, y=y_pos, width=60, height=25)
    btn_subsets_none = tk.Button(win, text="None", command=btn_subsets_none_onclick, state='disabled')
    btn_subsets_none.place(x=70, y=y_pos, width=60, height=25)
    lbl_subsets_filter = tk.Label(win, text="Filter:", anchor='e', state='disabled')
    lbl_subsets_filter.place(x=140, y=y_pos + 2, width=100, height=20)
    txt_subsets_filter = tk.Entry(win, textvariable=subset_filter, state='disabled')
    txt_subsets_filter.place(x=240, y=y_pos, width=window_width - 268, height=25)
    txt_subsets_filter.bind('<KeyRelease>', txt_subsets_filter_onkeyrelease)

    y_pos += 40
    tk.Label(win, text="Configuration:", anchor='w').place(x=5, y=y_pos, width=window_width - 10, height=20)
    y_pos += 20 + 5
    tk.Button(win, text="Import from JSON file...", command=config_preprocessing_callback).place(x=10, y=y_pos, width=window_width - 20, height=26)
    y_pos += 30 + 5
    config_btn_width = (window_width - 10 - 10 - 10) / 2
    tk.Button(win, text="Preprocessing", command=config_preprocessing_callback).place(x=10, y=y_pos, width=config_btn_width, height=28)
    tk.Button(win, text="Trials and channels", command=config_preprocessing_callback).place(x=10 + config_btn_width + 10, y=y_pos, width=config_btn_width, height=28)
    y_pos += 28
    tk.Button(win, text="Detection & Metrics", command=config_preprocessing_callback).place(x=10, y=y_pos, width=config_btn_width, height=28)
    tk.Button(win, text="Visualizations", command=config_preprocessing_callback).place(x=10 + config_btn_width + 10, y=y_pos, width=config_btn_width, height=28)
    #y_pos += 40 + 2
    #chk_apply_validator = tk.Checkbutton(win, text='Apply BIDS validator', anchor="w", variable=apply_bids_validator, onvalue=1, offvalue=0)
    #chk_apply_validator.place(x=20, y=y_pos, width=window_width - 30, height=20)

    y_pos += 45 + 2
    tk.Label(win, text="Process:", anchor='w').place(x=5, y=y_pos, width=window_width - 10, height=20)
    y_pos += 20 + 2
    btn_process = tk.Button(win, text="Start", command=btn_process_start_onclick, state='disabled')
    btn_process.place(x=10, y=y_pos, width=window_width - 20, height=40)
    y_pos += 40 + 2
    txt_console = tk.Text(win)
    scr_subsets = tk.Scrollbar(win, orient='vertical')
    txt_console.place(x=12, y=y_pos, width=window_width - 42, height=120)
    scr_subsets.place(x=window_width - 33, y=y_pos + 3, width=20, height=120 - 6)
    txt_console.config(yscrollcommand=scr_subsets.set)
    scr_subsets.config(command=txt_console.yview)
    y_pos += 120 + 2
    subject_pb = ttk.Progressbar(win, orient='horizontal', mode='determinate')
    subject_pb.place(x=15, y=y_pos, width=window_width - 30, height=30)

    # open window
    win.mainloop()
    exit()


if __name__ == "__main__":
    open_gui()
