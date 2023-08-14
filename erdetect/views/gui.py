"""
Evoked response detection - GUI entry-point
=====================================================
GUI entry-point python script for the automatic detection of evoked responses in CCEP data.


Copyright 2023, Max van den Boom (Multimodal Neuroimaging Lab, Mayo Clinic, Rochester MN)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import os
import logging
import threading

from ieegprep.fileio.IeegDataReader import VALID_FORMAT_EXTENSIONS
from ieegprep.bids.data_structure import list_bids_datasets
from ieegprep.utils.misc import is_number
from erdetect.core.config import load_config, get as cfg, set as cfg_set, rem as cfg_rem, create_default_config
from erdetect._erdetect import process_subset, print_config

def open_gui(init_input_directory=None, init_output_directory=None):
    """

    """

    # Python might not be configured for tk, so by importing it here, the
    # rest (functions and command-line wrapper) can run without trouble
    import tkinter as tk
    import tkinter.ttk as ttk
    from tkinter import filedialog, messagebox

    BTN_TOGGLE_ON_BACKGROUND = "#8296FD"

    # custom callbacks
    def _txt_double_input_validate(char, entry_value):
        return is_number(entry_value) or entry_value in ('-', '-.', '.', ',', '')

    def _txt_double_pos_input_validate(char, entry_value):
        if char == '-':
            return False
        return is_number(entry_value) or entry_value in ('.', ',', '')

    def _txt_integer_pos_input_validate(char, entry_value):
        if char in ('.', '-'):
            return False
        return is_number(entry_value) or entry_value == ''

    def _update_combo_losefocus(self):
        self.widget.master.focus()

    def _btn_toggle(button):
        if button.config('relief')[-1] == 'sunken':
            _btn_set_state(button, False, False)
        else:
            _btn_set_state(button, False, True)

    def _btn_set_state(button, disabled, on):
        button.config(relief='sunken' if on else 'raised')
        if not disabled and on:
            button.config(background=BTN_TOGGLE_ON_BACKGROUND, foreground='white')
        else:
            button.config(background=win['background'], foreground='black')
        button.configure(state='disabled' if disabled else 'normal')

    def _create_channel_types_lst(btn_ecog, btn_seeg, btn_dbs):
        lst_channel_types = []
        if btn_ecog.config('relief')[-1] == 'sunken':
            lst_channel_types.append('ECOG')
        if btn_seeg.config('relief')[-1] == 'sunken':
            lst_channel_types.append('SEEG')
        if btn_dbs.config('relief')[-1] == 'sunken':
            lst_channel_types.append('DBS')
        return lst_channel_types


    #
    # pre-process configuration dialog
    #
    class PreprocessDialog(object):

        reref_options = {'CAR': 'Common Average Re-refencing (CAR)', 'CAR_headbox': 'Common Average Re-refencing (CAR) per headbox'}
        reref_options_values = {v: k for k, v in reref_options.items()}

        line_noise_removal_options = {'JSON': 'JSON/sidecar (from *_ieeg.json files)', '50': '50Hz', '60': '60Hz'}
        line_noise_removal_options_values = {v: k for k, v in line_noise_removal_options.items()}

        def _update_early_reref_controls(self):
            new_state = 'normal' if self.early_reref.get() else 'disabled'
            self.lbl_early_reref_method.configure(state=new_state)
            self.cmb_early_reref_method.configure(state=new_state)
            self.lbl_early_reref_excl_epoch.configure(state=new_state)
            self.txt_early_reref_excl_epoch_start.configure(state=new_state)
            self.lbl_early_reref_excl_epoch_range.configure(state=new_state)
            self.txt_early_reref_excl_epoch_end.configure(state=new_state)
            self.lbl_early_reref_channel_types.configure(state=new_state)
            _btn_set_state(self.btn_early_reref_channel_types_ecog, new_state == 'disabled', self.btn_early_reref_channel_types_ecog.config('relief')[-1] == 'sunken')
            _btn_set_state(self.btn_early_reref_channel_types_seeg, new_state == 'disabled', self.btn_early_reref_channel_types_seeg.config('relief')[-1] == 'sunken')
            _btn_set_state(self.btn_early_reref_channel_types_dbs, new_state == 'disabled', self.btn_early_reref_channel_types_dbs.config('relief')[-1] == 'sunken')

        def _update_line_noise_controls(self):
            new_state = 'normal' if self.line_noise_removal.get() else 'disabled'
            self.lbl_line_noise_frequency.configure(state=new_state)
            self.cmb_line_noise_frequency.configure(state=new_state)

        def _update_late_reref_controls(self):
            new_state = 'normal' if self.late_reref.get() else 'disabled'
            self.lbl_late_reref_method.configure(state=new_state)
            self.cmb_late_reref_method.configure(state=new_state)
            self.lbl_late_reref_excl_epoch.configure(state=new_state)
            self.txt_late_reref_excl_epoch_start.configure(state=new_state)
            self.lbl_late_reref_excl_epoch_range.configure(state=new_state)
            self.txt_late_reref_excl_epoch_end.configure(state=new_state)
            self.lbl_late_reref_channel_types.configure(state=new_state)
            _btn_set_state(self.btn_late_reref_channel_types_ecog, new_state == 'disabled', self.btn_late_reref_channel_types_ecog.config('relief')[-1] == 'sunken')
            _btn_set_state(self.btn_late_reref_channel_types_seeg, new_state == 'disabled', self.btn_late_reref_channel_types_seeg.config('relief')[-1] == 'sunken')
            _btn_set_state(self.btn_late_reref_channel_types_dbs, new_state == 'disabled', self.btn_late_reref_channel_types_dbs.config('relief')[-1] == 'sunken')

            new_CAR_state = 'normal' if self.late_reref.get() and self.reref_options_values[self.late_reref_method.get()] in ('CAR', 'CAR_headbox') else 'disabled'
            self.lbl_late_reref_CAR_variance.configure(state=new_CAR_state)
            self.chk_late_reref_CAR_variance.configure(state=new_CAR_state)

            new_CAR_variance_state = 'normal' if new_CAR_state == 'normal' and self.late_reref_CAR_variance_enabled.get() else 'disabled'
            self.lbl_late_reref_CAR_variance_quantile.configure(state=new_CAR_variance_state)
            self.txt_late_reref_CAR_variance_quantile.configure(state=new_CAR_variance_state)
            self.lbl_late_reref_CAR_variance_quantile2.configure(state=new_CAR_variance_state)

        def __init__(self, parent):
            pd_window_height = 520
            pd_window_width = 630

            # retrieve values from config
            self.highpass = tk.IntVar(value=cfg('preprocess', 'high_pass'))
            self.early_reref = tk.IntVar(value=cfg('preprocess', 'early_re_referencing', 'enabled'))
            self.early_reref_method = tk.StringVar(value=self.reref_options[str(cfg('preprocess', 'early_re_referencing', 'method'))])
            self.early_reref_excl_epoch_start = tk.DoubleVar(value=cfg('preprocess', 'early_re_referencing', 'stim_excl_epoch')[0])
            self.early_reref_excl_epoch_end = tk.DoubleVar(value=cfg('preprocess', 'early_re_referencing', 'stim_excl_epoch')[1])

            if cfg('preprocess', 'line_noise_removal').lower() == 'off':
                self.line_noise_removal = tk.IntVar(value=0)
                self.line_noise_frequency = tk.StringVar(value=self.line_noise_removal_options['JSON'])
            else:
                self.line_noise_removal = tk.IntVar(value=1)
                if cfg('preprocess', 'line_noise_removal').lower() == 'json':
                    self.line_noise_frequency = tk.StringVar(value=self.line_noise_removal_options['JSON'])
                else:
                    self.line_noise_frequency = tk.StringVar(value=self.line_noise_removal_options[str(cfg('preprocess', 'line_noise_removal'))])

            self.late_reref = tk.IntVar(value=cfg('preprocess', 'late_re_referencing', 'enabled'))
            self.late_reref_method = tk.StringVar(value=self.reref_options[str(cfg('preprocess', 'late_re_referencing', 'method'))])
            self.late_reref_excl_epoch_start = tk.DoubleVar(value=cfg('preprocess', 'late_re_referencing', 'stim_excl_epoch')[0])
            self.late_reref_excl_epoch_end = tk.DoubleVar(value=cfg('preprocess', 'late_re_referencing', 'stim_excl_epoch')[1])
            self.late_reref_CAR_variance_enabled = tk.IntVar(value=cfg('preprocess', 'late_re_referencing', 'CAR_by_variance') != -1)
            if self.late_reref_CAR_variance_enabled.get():
                self.late_reref_CAR_variance_quantile = tk.DoubleVar(value=cfg('preprocess', 'late_re_referencing', 'CAR_by_variance'))
            else:
                self.late_reref_CAR_variance_quantile = tk.DoubleVar(value=0.2)


            #
            # elements
            #

            self.root = tk.Toplevel(parent)
            self.root.title('Preprocessing')
            self.root.geometry("{}x{}+{}+{}".format(pd_window_width, pd_window_height,
                                      int((win.winfo_screenwidth() / 2) - (pd_window_width / 2)),
                                      int((win.winfo_screenheight() / 2) - (pd_window_height / 2))))
            self.root.resizable(False, False)

            #
            pd_y_pos = 15
            self.chk_highpass = tk.Checkbutton(self.root, text='High pass filtering (0.50Hz)', anchor="w", variable=self.highpass, onvalue=1, offvalue=0)
            self.chk_highpass.place(x=10, y=pd_y_pos, width=pd_window_width, height=30)
            pd_y_pos += 30
            self.chk_early_reref = tk.Checkbutton(self.root, text='Early re-referencing:', anchor="w", variable=self.early_reref, onvalue=1, offvalue=0, command=self._update_early_reref_controls)
            self.chk_early_reref.place(x=10, y=pd_y_pos, width=pd_window_width, height=30)
            pd_y_pos += 32
            early_reref_state = 'normal' if self.early_reref.get() else 'disabled'
            self.lbl_early_reref_method = tk.Label(self.root, text="Method", anchor='e', state=early_reref_state)
            self.lbl_early_reref_method.place(x=5, y=pd_y_pos + 2, width=245, height=20)
            self.cmb_early_reref_method = ttk.Combobox(self.root, textvariable=self.early_reref_method, values=list(self.reref_options_values.keys()), state=early_reref_state)
            self.cmb_early_reref_method.bind("<Key>", lambda e: "break")
            self.cmb_early_reref_method.bind("<<ComboboxSelected>>", _update_combo_losefocus)
            self.cmb_early_reref_method.bind("<FocusIn>", _update_combo_losefocus)
            self.cmb_early_reref_method.place(x=260, y=pd_y_pos, width=350, height=25)
            pd_y_pos += 32
            self.lbl_early_reref_excl_epoch = tk.Label(self.root, text="Stim. channel exclusion window", anchor='e', state=early_reref_state)
            self.lbl_early_reref_excl_epoch.place(x=5, y=pd_y_pos + 2, width=245, height=20)
            self.txt_early_reref_excl_epoch_start = ttk.Entry(self.root, textvariable=self.early_reref_excl_epoch_start, state=early_reref_state, justify='center', validate = 'key', validatecommand=(self.root.register(_txt_double_input_validate), '%S', '%P'))
            self.txt_early_reref_excl_epoch_start.place(x=260, y=pd_y_pos, width=70, height=25)
            self.lbl_early_reref_excl_epoch_range = tk.Label(self.root, text="-", state=early_reref_state)
            self.lbl_early_reref_excl_epoch_range.place(x=335, y=pd_y_pos, width=30, height=25)
            self.txt_early_reref_excl_epoch_end = ttk.Entry(self.root, textvariable=self.early_reref_excl_epoch_end, state=early_reref_state, justify='center', validate = 'key', validatecommand=(self.root.register(_txt_double_input_validate), '%S', '%P'))
            self.txt_early_reref_excl_epoch_end.place(x=370, y=pd_y_pos, width=70, height=25)
            pd_y_pos += 32
            self.lbl_early_reref_channel_types = tk.Label(self.root, text="Included channel-types", anchor='e', state=early_reref_state)
            self.lbl_early_reref_channel_types.place(x=5, y=pd_y_pos + 2, width=245, height=20)
            self.btn_early_reref_channel_types_ecog = tk.Button(self.root, text="ECoG", command=lambda:_btn_toggle(self.btn_early_reref_channel_types_ecog), state=early_reref_state)
            self.btn_early_reref_channel_types_ecog.place(x=260, y=pd_y_pos, width=65, height=25)
            self.btn_early_reref_channel_types_seeg = tk.Button(self.root, text="SEEG", command=lambda:_btn_toggle(self.btn_early_reref_channel_types_seeg), state=early_reref_state)
            self.btn_early_reref_channel_types_seeg.place(x=326, y=pd_y_pos, width=65, height=25)
            self.btn_early_reref_channel_types_dbs = tk.Button(self.root, text="DBS", command=lambda:_btn_toggle(self.btn_early_reref_channel_types_dbs), state=early_reref_state)
            self.btn_early_reref_channel_types_dbs.place(x=392, y=pd_y_pos, width=65, height=25)
            early_reref_channel_types = [x.lower() for x in cfg('preprocess', 'early_re_referencing', 'channel_types')]
            _btn_set_state(self.btn_early_reref_channel_types_ecog, early_reref_state == 'disabled', 'ecog' in early_reref_channel_types)
            _btn_set_state(self.btn_early_reref_channel_types_seeg, early_reref_state == 'disabled', 'seeg' in early_reref_channel_types)
            _btn_set_state(self.btn_early_reref_channel_types_dbs, early_reref_state == 'disabled', 'dbs' in early_reref_channel_types)
            pd_y_pos += 42

            self.chk_line_noise = tk.Checkbutton(self.root, text='Line-noise removal:', anchor="w", variable=self.line_noise_removal, onvalue=1, offvalue=0, command=self._update_line_noise_controls)
            self.chk_line_noise.place(x=10, y=pd_y_pos, width=pd_window_width, height=30)
            pd_y_pos += 32
            line_noise_state = 'normal' if self.line_noise_removal.get() else 'disabled'
            self.lbl_line_noise_frequency = tk.Label(self.root, text="Frequency", anchor='e', state=line_noise_state)
            self.lbl_line_noise_frequency.place(x=5, y=pd_y_pos + 2, width=245, height=20)
            self.cmb_line_noise_frequency = ttk.Combobox(self.root, textvariable=self.line_noise_frequency, values=list(self.line_noise_removal_options_values.keys()), state=line_noise_state)
            self.cmb_line_noise_frequency.bind("<Key>", lambda e: "break")
            self.cmb_line_noise_frequency.bind("<<ComboboxSelected>>", _update_combo_losefocus)
            self.cmb_line_noise_frequency.bind("<FocusIn>", _update_combo_losefocus)
            self.cmb_line_noise_frequency.place(x=260, y=pd_y_pos, width=350, height=25)
            pd_y_pos += 42

            self.chk_late_reref = tk.Checkbutton(self.root, text='Late re-referencing:', anchor="w", variable=self.late_reref, onvalue=1, offvalue=0, command=self._update_late_reref_controls)
            self.chk_late_reref.place(x=10, y=pd_y_pos, width=pd_window_width, height=30)
            pd_y_pos += 32
            late_reref_state = 'normal' if self.late_reref.get() else 'disabled'
            self.lbl_late_reref_method = tk.Label(self.root, text="Method", anchor='e', state=late_reref_state)
            self.lbl_late_reref_method.place(x=5, y=pd_y_pos + 2, width=245, height=20)
            self.cmb_late_reref_method = ttk.Combobox(self.root, textvariable=self.late_reref_method, values=list(self.reref_options_values.keys()), state=late_reref_state)
            self.cmb_late_reref_method.bind("<Key>", lambda e: "break")
            self.cmb_late_reref_method.bind("<<ComboboxSelected>>", _update_combo_losefocus)
            self.cmb_late_reref_method.bind("<FocusIn>", _update_combo_losefocus)
            self.cmb_late_reref_method.place(x=260, y=pd_y_pos, width=350, height=25)
            pd_y_pos += 32
            self.lbl_late_reref_excl_epoch = tk.Label(self.root, text="Stim. channel exclusion window", anchor='e', state=late_reref_state)
            self.lbl_late_reref_excl_epoch.place(x=5, y=pd_y_pos + 2, width=245, height=20)
            self.txt_late_reref_excl_epoch_start = ttk.Entry(self.root, textvariable=self.late_reref_excl_epoch_start, state=late_reref_state, justify='center', validate = 'key', validatecommand=(self.root.register(_txt_double_input_validate), '%S', '%P'))
            self.txt_late_reref_excl_epoch_start.place(x=260, y=pd_y_pos, width=70, height=25)
            self.lbl_late_reref_excl_epoch_range = tk.Label(self.root, text="-", state=late_reref_state)
            self.lbl_late_reref_excl_epoch_range.place(x=335, y=pd_y_pos, width=30, height=25)
            self.txt_late_reref_excl_epoch_end = ttk.Entry(self.root, textvariable=self.late_reref_excl_epoch_end, state=late_reref_state, justify='center', validate = 'key', validatecommand=(self.root.register(_txt_double_input_validate), '%S', '%P'))
            self.txt_late_reref_excl_epoch_end.place(x=370, y=pd_y_pos, width=70, height=25)
            pd_y_pos += 32
            self.lbl_late_reref_channel_types = tk.Label(self.root, text="Included channel-types", anchor='e', state=late_reref_state)
            self.lbl_late_reref_channel_types.place(x=5, y=pd_y_pos + 2, width=245, height=20)
            self.btn_late_reref_channel_types_ecog = tk.Button(self.root, text="ECoG", command=lambda:_btn_toggle(self.btn_late_reref_channel_types_ecog), state=late_reref_state)
            self.btn_late_reref_channel_types_ecog.place(x=260, y=pd_y_pos, width=65, height=25)
            self.btn_late_reref_channel_types_seeg = tk.Button(self.root, text="SEEG", command=lambda:_btn_toggle(self.btn_late_reref_channel_types_seeg), state=late_reref_state)
            self.btn_late_reref_channel_types_seeg.place(x=326, y=pd_y_pos, width=65, height=25)
            self.btn_late_reref_channel_types_dbs = tk.Button(self.root, text="DBS", command=lambda:_btn_toggle(self.btn_late_reref_channel_types_dbs), state=late_reref_state)
            self.btn_late_reref_channel_types_dbs.place(x=392, y=pd_y_pos, width=65, height=25)
            late_reref_channel_types = [x.lower() for x in cfg('preprocess', 'late_re_referencing', 'channel_types')]
            _btn_set_state(self.btn_late_reref_channel_types_ecog, late_reref_state == 'disabled', 'ecog' in late_reref_channel_types)
            _btn_set_state(self.btn_late_reref_channel_types_seeg, late_reref_state == 'disabled', 'seeg' in late_reref_channel_types)
            _btn_set_state(self.btn_late_reref_channel_types_dbs, late_reref_state == 'disabled', 'dbs' in late_reref_channel_types)
            pd_y_pos += 32

            late_reref_CAR_state = 'normal' if self.late_reref.get() and self.reref_options_values[self.late_reref_method.get()] in ('CAR', 'CAR_headbox') else 'disabled'
            self.lbl_late_reref_CAR_variance = tk.Label(self.root, text="Select channels by trial variance", anchor='e', state=late_reref_CAR_state)
            self.lbl_late_reref_CAR_variance.place(x=5, y=pd_y_pos + 2, width=245, height=20)
            self.chk_late_reref_CAR_variance = tk.Checkbutton(self.root, text='', anchor="w", variable=self.late_reref_CAR_variance_enabled, onvalue=1, offvalue=0, command=self._update_late_reref_controls, state=late_reref_CAR_state)
            self.chk_late_reref_CAR_variance.place(x=256, y=pd_y_pos, width=pd_window_width, height=30)

            late_reref_CAR_variance_state = 'normal' if late_reref_CAR_state == 'normal' and self.late_reref_CAR_variance_enabled.get() else 'disabled'
            self.lbl_late_reref_CAR_variance_quantile = tk.Label(self.root, text="within", anchor='w', state=late_reref_CAR_variance_state)
            self.lbl_late_reref_CAR_variance_quantile.place(x=285, y=pd_y_pos + 2, width=100, height=20)
            self.txt_late_reref_CAR_variance_quantile = ttk.Entry(self.root, textvariable=self.late_reref_CAR_variance_quantile, state=late_reref_CAR_variance_state, justify='center', validate = 'key', validatecommand=(self.root.register(_txt_double_pos_input_validate), '%S', '%P'))
            self.txt_late_reref_CAR_variance_quantile.place(x=340, y=pd_y_pos, width=60, height=25)
            self.lbl_late_reref_CAR_variance_quantile2 = tk.Label(self.root, text="quantile", anchor='w', state=late_reref_CAR_variance_state)
            self.lbl_late_reref_CAR_variance_quantile2.place(x=410, y=pd_y_pos + 2, width=100, height=20)

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
            lst_early_reref_channel_types = tuple(_create_channel_types_lst(self.btn_early_reref_channel_types_ecog, self.btn_early_reref_channel_types_seeg, self.btn_early_reref_channel_types_dbs))
            lst_late_reref_channel_types = tuple(_create_channel_types_lst(self.btn_late_reref_channel_types_ecog, self.btn_late_reref_channel_types_seeg, self.btn_late_reref_channel_types_dbs))

            # check input values
            if self.early_reref.get():
                if self.early_reref_excl_epoch_end.get() < self.early_reref_excl_epoch_start.get():
                    messagebox.showerror(title='Invalid input', message='Invalid window (start and end values) for the early re-referencing stim exclusion setting. '
                                                                        'The given end-point (' + str(self.early_reref_excl_epoch_end.get()) + ') lies before the start-point (' + str(self.early_reref_excl_epoch_start.get()) + ')')
                    self.txt_early_reref_excl_epoch_start.focus()
                    self.txt_early_reref_excl_epoch_start.select_range(0, tk.END)
                    return
                if len(lst_early_reref_channel_types) == 0:
                    messagebox.showerror(title='Invalid input', message='At least one channel type should be enabled for the early re-referencing included channel-types setting.')
                    return
            if self.late_reref.get():
                if self.late_reref_excl_epoch_end.get() < self.late_reref_excl_epoch_start.get():
                    messagebox.showerror(title='Invalid input', message='Invalid window (start and end values) for the late re-referencing stim exclusion setting. '
                                                                        'The given end-point (' + str(self.late_reref_excl_epoch_end.get()) + ') lies before the start-point (' + str(self.late_reref_excl_epoch_start.get()) + ')')
                    self.txt_late_reref_excl_epoch_start.focus()
                    self.txt_late_reref_excl_epoch_start.select_range(0, tk.END)
                    return
                if len(lst_late_reref_channel_types) == 0:
                    messagebox.showerror(title='Invalid input', message='At least one channel type should be enabled for the late re-referencing included channel-types setting.')
                    return
                if self.late_reref_CAR_variance_enabled.get():
                    if self.late_reref_CAR_variance_quantile.get() < 0 or self.late_reref_CAR_variance_quantile.get() > 1:
                        messagebox.showerror(title='Invalid input', message='Invalid quantile value for channel selection by trial variance, the value should be between 0 and 1')
                        self.txt_late_reref_CAR_variance_quantile.focus()
                        self.txt_late_reref_CAR_variance_quantile.select_range(0, tk.END)
                        return

            # update config
            cfg_set(self.highpass.get(), 'preprocess', 'high_pass')
            cfg_set(self.early_reref.get(), 'preprocess', 'early_re_referencing', 'enabled')
            cfg_set(self.reref_options_values[self.early_reref_method.get()], 'preprocess', 'early_re_referencing', 'method')
            if self.early_reref.get():
                cfg_set((self.early_reref_excl_epoch_start.get(), self.early_reref_excl_epoch_end.get()), 'preprocess', 'early_re_referencing', 'stim_excl_epoch')
            cfg_set(lst_early_reref_channel_types, 'preprocess', 'early_re_referencing', 'channel_types')

            if self.line_noise_removal.get():
                cfg_set(self.line_noise_removal_options_values[self.line_noise_frequency.get()], 'preprocess', 'line_noise_removal')
            else:
                cfg_set('off', 'preprocess', 'line_noise_removal')

            cfg_set(self.late_reref.get(), 'preprocess', 'late_re_referencing', 'enabled')
            cfg_set(self.reref_options_values[self.late_reref_method.get()], 'preprocess', 'late_re_referencing', 'method')
            cfg_set(lst_late_reref_channel_types, 'preprocess', 'late_re_referencing', 'channel_types')
            if self.late_reref.get():
                cfg_set((self.late_reref_excl_epoch_start.get(), self.late_reref_excl_epoch_end.get()), 'preprocess', 'late_re_referencing', 'stim_excl_epoch')
                if self.late_reref_CAR_variance_enabled.get():
                    cfg_set(self.late_reref_CAR_variance_quantile.get(), 'preprocess', 'late_re_referencing', 'CAR_by_variance')
                else:
                    cfg_set(-1, 'preprocess', 'late_re_referencing', 'CAR_by_variance')

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
            self.early_reref_method.set(self.reref_options[config_defaults['preprocess']['early_re_referencing']['method']])
            self.early_reref_excl_epoch_start.set(config_defaults['preprocess']['early_re_referencing']['stim_excl_epoch'][0])
            self.early_reref_excl_epoch_end.set(config_defaults['preprocess']['early_re_referencing']['stim_excl_epoch'][1])
            early_reref_channel_types = [x.lower() for x in config_defaults['preprocess']['early_re_referencing']['channel_types']]
            _btn_set_state(self.btn_early_reref_channel_types_ecog, self.early_reref.get() == 0, 'ecog' in early_reref_channel_types)
            _btn_set_state(self.btn_early_reref_channel_types_seeg, self.early_reref.get() == 0, 'seeg' in early_reref_channel_types)
            _btn_set_state(self.btn_early_reref_channel_types_dbs, self.early_reref.get() == 0, 'dbs' in early_reref_channel_types)

            if config_defaults['preprocess']['line_noise_removal'].lower() == 'off':
                self.line_noise_removal.set(0)
                self.line_noise_frequency.set(self.line_noise_removal_options['JSON'])
            else:
                self.line_noise_removal.set(1)
                self.line_noise_frequency.set(self.line_noise_removal_options[config_defaults['preprocess']['line_noise_removal']])

            self.late_reref.set(config_defaults['preprocess']['late_re_referencing']['enabled'])
            self.late_reref_method.set(self.reref_options[config_defaults['preprocess']['late_re_referencing']['method']])
            self.late_reref_excl_epoch_start.set(config_defaults['preprocess']['late_re_referencing']['stim_excl_epoch'][0])
            self.late_reref_excl_epoch_end.set(config_defaults['preprocess']['late_re_referencing']['stim_excl_epoch'][1])
            late_reref_channel_types = [x.lower() for x in config_defaults['preprocess']['late_re_referencing']['channel_types']]
            _btn_set_state(self.btn_late_reref_channel_types_ecog, self.late_reref.get() == 0, 'ecog' in late_reref_channel_types)
            _btn_set_state(self.btn_late_reref_channel_types_seeg, self.late_reref.get() == 0, 'seeg' in late_reref_channel_types)
            _btn_set_state(self.btn_late_reref_channel_types_dbs, self.late_reref.get() == 0, 'dbs' in late_reref_channel_types)
            self.late_reref_CAR_variance_enabled.set(config_defaults['preprocess']['late_re_referencing']['CAR_by_variance'] != -1)
            if config_defaults['preprocess']['late_re_referencing']['CAR_by_variance'] != -1:
                self.late_reref_CAR_variance_quantile.set(config_defaults['preprocess']['late_re_referencing']['CAR_by_variance'])
            else:
                self.late_reref_CAR_variance_quantile.set(.2)

            self._update_early_reref_controls()
            self._update_line_noise_controls()
            self._update_late_reref_controls()


    #
    # trials & channel-types configuration dialog
    #
    class TrialsAndChannelTypesDialog(object):

        out_of_bounds_options = {'first_last_only': 'Allow only for the first and last trials to be out-of-bounds', 'allow': 'Allow for any trial to be out-of-bounds', 'error': 'Do not allow any trial to be out-of-bounds'}
        out_of_bounds_options_values = {v: k for k, v in out_of_bounds_options.items()}

        baseline_norm_options = {'median': 'By baseline median', 'mean': 'By baseline mean', 'none': 'No normalization'}
        baseline_norm_options_values = {v: k for k, v in baseline_norm_options.items()}

        def __init__(self, parent):
            pd_window_height = 450
            pd_window_width = 700

            # retrieve values from config
            self.trial_epoch_start = tk.DoubleVar(value=cfg('trials', 'trial_epoch')[0])
            self.trial_epoch_end = tk.DoubleVar(value=cfg('trials', 'trial_epoch')[1])
            self.out_of_bounds_handling = tk.StringVar(value=self.out_of_bounds_options[str(cfg('trials', 'out_of_bounds_handling'))])
            self.baseline_norm = tk.StringVar(value=self.baseline_norm_options[str(cfg('trials', 'baseline_norm'))])
            self.baseline_epoch_start = tk.DoubleVar(value=cfg('trials', 'baseline_epoch')[0])
            self.baseline_epoch_end = tk.DoubleVar(value=cfg('trials', 'baseline_epoch')[1])
            self.concat_bidir_stimpairs = tk.IntVar(value=cfg('trials', 'concat_bidirectional_pairs'))
            self.min_stimpair_trials = tk.IntVar(value=cfg('trials', 'minimum_stimpair_trials'))

            #
            # elements
            #
            self.root = tk.Toplevel(parent)
            self.root.title('Trials and Channel-types')
            self.root.geometry("{}x{}+{}+{}".format(pd_window_width, pd_window_height,
                                      int((win.winfo_screenwidth() / 2) - (pd_window_width / 2)),
                                      int((win.winfo_screenheight() / 2) - (pd_window_height / 2))))
            self.root.resizable(False, False)

            #
            pd_y_pos = 15
            self.lbl_trial_epoch = tk.Label(self.root, text="Trial window", anchor='e')
            self.lbl_trial_epoch.place(x=5, y=pd_y_pos + 2, width=245, height=20)
            self.txt_trial_epoch_start = ttk.Entry(self.root, textvariable=self.trial_epoch_start, justify='center', validate = 'key', validatecommand=(self.root.register(_txt_double_input_validate), '%S', '%P'))
            self.txt_trial_epoch_start.place(x=260, y=pd_y_pos, width=70, height=25)
            self.lbl_trial_range = tk.Label(self.root, text="-")
            self.lbl_trial_range.place(x=335, y=pd_y_pos, width=30, height=25)
            self.txt_trial_end = ttk.Entry(self.root, textvariable=self.trial_epoch_end, justify='center', validate = 'key', validatecommand=(self.root.register(_txt_double_input_validate), '%S', '%P'))
            self.txt_trial_end.place(x=370, y=pd_y_pos, width=70, height=25)
            pd_y_pos += 32
            self.lbl_out_of_bounds_handling = tk.Label(self.root, text="Out-of-bounds handling", anchor='e')
            self.lbl_out_of_bounds_handling.place(x=5, y=pd_y_pos + 2, width=245, height=20)
            self.cmb_out_of_bounds_handling = ttk.Combobox(self.root, textvariable=self.out_of_bounds_handling, values=list(self.out_of_bounds_options_values.keys()))
            self.cmb_out_of_bounds_handling.bind("<Key>", lambda e: "break")
            self.cmb_out_of_bounds_handling.bind("<<ComboboxSelected>>", _update_combo_losefocus)
            self.cmb_out_of_bounds_handling.bind("<FocusIn>", _update_combo_losefocus)
            self.cmb_out_of_bounds_handling.place(x=260, y=pd_y_pos, width=400, height=25)
            pd_y_pos += 50
            self.lbl_baseline_norm = tk.Label(self.root, text="Trial normalization", anchor='e')
            self.lbl_baseline_norm.place(x=5, y=pd_y_pos + 2, width=245, height=20)
            self.cmb_baseline_norm = ttk.Combobox(self.root, textvariable=self.baseline_norm, values=list(self.baseline_norm_options_values.keys()))
            self.cmb_baseline_norm.bind("<Key>", lambda e: "break")
            self.cmb_baseline_norm.bind("<<ComboboxSelected>>", _update_combo_losefocus)
            self.cmb_baseline_norm.bind("<FocusIn>", _update_combo_losefocus)
            self.cmb_baseline_norm.place(x=260, y=pd_y_pos, width=250, height=25)
            pd_y_pos += 32
            self.lbl_baseline_epoch = tk.Label(self.root, text="Baseline window", anchor='e')
            self.lbl_baseline_epoch.place(x=5, y=pd_y_pos + 2, width=245, height=20)
            self.txt_baseline_epoch_start = ttk.Entry(self.root, textvariable=self.baseline_epoch_start, justify='center', validate = 'key', validatecommand=(self.root.register(_txt_double_input_validate), '%S', '%P'))
            self.txt_baseline_epoch_start.place(x=260, y=pd_y_pos, width=70, height=25)
            self.lbl_baseline_range = tk.Label(self.root, text="-")
            self.lbl_baseline_range.place(x=335, y=pd_y_pos, width=30, height=25)
            self.txt_baseline_end = ttk.Entry(self.root, textvariable=self.baseline_epoch_end, justify='center', validate = 'key', validatecommand=(self.root.register(_txt_double_input_validate), '%S', '%P'))
            self.txt_baseline_end.place(x=370, y=pd_y_pos, width=70, height=25)
            pd_y_pos += 44
            self.lbl_concat_bidir_stimpairs = tk.Label(self.root, text="Concatenate bidirectional stim-pairs", anchor='e')
            self.lbl_concat_bidir_stimpairs.place(x=5, y=pd_y_pos + 2, width=245, height=20)
            self.chk_concat_bidir_stimpairs = tk.Checkbutton(self.root, text='', anchor="w", variable=self.concat_bidir_stimpairs, onvalue=1, offvalue=0)
            self.chk_concat_bidir_stimpairs.place(x=256, y=pd_y_pos, width=pd_window_width, height=30)
            pd_y_pos += 32
            self.lbl_min_stimpair_trials = tk.Label(self.root, text="Minimal number trials per stim-pair", anchor='e')
            self.lbl_min_stimpair_trials.place(x=5, y=pd_y_pos + 2, width=245, height=20)
            self.txt_min_stimpair_trials = ttk.Entry(self.root, textvariable=self.min_stimpair_trials, justify='center', validate = 'key', validatecommand=(self.root.register(_txt_integer_pos_input_validate), '%S', '%P'))
            self.txt_min_stimpair_trials.place(x=260, y=pd_y_pos, width=40, height=25)
            pd_y_pos += 50
            self.lbl_measured_channel_types = tk.Label(self.root, text="Include types as measured electrodes:", anchor='e')
            self.lbl_measured_channel_types.place(x=5, y=pd_y_pos + 2, width=245, height=20)
            self.btn_measured_channel_types_ecog = tk.Button(self.root, text="ECoG", command=lambda:_btn_toggle(self.btn_measured_channel_types_ecog))
            self.btn_measured_channel_types_ecog.place(x=260, y=pd_y_pos, width=65, height=25)
            self.btn_measured_channel_types_seeg = tk.Button(self.root, text="SEEG", command=lambda:_btn_toggle(self.btn_measured_channel_types_seeg))
            self.btn_measured_channel_types_seeg.place(x=326, y=pd_y_pos, width=65, height=25)
            self.btn_measured_channel_types_dbs = tk.Button(self.root, text="DBS", command=lambda:_btn_toggle(self.btn_measured_channel_types_dbs))
            self.btn_measured_channel_types_dbs.place(x=392, y=pd_y_pos, width=65, height=25)
            measured_channel_types = [x.lower() for x in cfg('channels', 'measured_types')]
            _btn_set_state(self.btn_measured_channel_types_ecog, False, 'ecog' in measured_channel_types)
            _btn_set_state(self.btn_measured_channel_types_seeg, False, 'seeg' in measured_channel_types)
            _btn_set_state(self.btn_measured_channel_types_dbs, False, 'dbs' in measured_channel_types)
            pd_y_pos += 32
            self.lbl_stim_channel_types = tk.Label(self.root, text="Include types for stimulation:", anchor='e')
            self.lbl_stim_channel_types.place(x=5, y=pd_y_pos + 2, width=245, height=20)
            self.btn_stim_channel_types_ecog = tk.Button(self.root, text="ECoG", command=lambda:_btn_toggle(self.btn_stim_channel_types_ecog))
            self.btn_stim_channel_types_ecog.place(x=260, y=pd_y_pos, width=65, height=25)
            self.btn_stim_channel_types_seeg = tk.Button(self.root, text="SEEG", command=lambda:_btn_toggle(self.btn_stim_channel_types_seeg))
            self.btn_stim_channel_types_seeg.place(x=326, y=pd_y_pos, width=65, height=25)
            self.btn_stim_channel_types_dbs = tk.Button(self.root, text="DBS", command=lambda:_btn_toggle(self.btn_stim_channel_types_dbs))
            self.btn_stim_channel_types_dbs.place(x=392, y=pd_y_pos, width=65, height=25)
            stim_channel_types = [x.lower() for x in cfg('channels', 'stim_types')]
            _btn_set_state(self.btn_stim_channel_types_ecog, False, 'ecog' in stim_channel_types)
            _btn_set_state(self.btn_stim_channel_types_seeg, False, 'seeg' in stim_channel_types)
            _btn_set_state(self.btn_stim_channel_types_dbs, False, 'dbs' in stim_channel_types)

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
            lst_measured_channel_types = tuple(_create_channel_types_lst(self.btn_measured_channel_types_ecog, self.btn_measured_channel_types_seeg, self.btn_measured_channel_types_dbs))
            lst_stim_channel_types = tuple(_create_channel_types_lst(self.btn_stim_channel_types_ecog, self.btn_stim_channel_types_seeg, self.btn_stim_channel_types_dbs))

            # check input values
            if self.trial_epoch_end.get() <= self.trial_epoch_start.get():
                messagebox.showerror(title='Invalid input', message='Invalid window (start and end values) for the trial epoch setting. '
                                                                    'The given end-point (' + str(self.trial_epoch_end.get()) + ') lies on or before the start-point (' + str(self.trial_epoch_start.get()) + ')')
                self.txt_trial_epoch_start.focus()
                self.txt_trial_epoch_start.select_range(0, tk.END)
                return
            if self.baseline_epoch_end.get() <= self.baseline_epoch_start.get():
                messagebox.showerror(title='Invalid input', message='Invalid window (start and end values) for the baseline epoch setting. '
                                                                    'The given end-point (' + str(self.baseline_epoch_end.get()) + ') lies on or before the start-point (' + str(self.baseline_epoch_start.get()) + ')')
                self.txt_baseline_epoch_start.focus()
                self.txt_baseline_epoch_start.select_range(0, tk.END)
                return
            if len(lst_measured_channel_types) == 0:
                messagebox.showerror(title='Invalid input', message='At least one channel type should be enabled for type of channels included for measurements setting.')
                return
            if len(lst_stim_channel_types) == 0:
                messagebox.showerror(title='Invalid input', message='At least one channel type should be enabled for type of channels included for stimulation setting.')
                return

            # update config
            cfg_set((self.trial_epoch_start.get(), self.trial_epoch_end.get()), 'trials', 'trial_epoch')
            cfg_set(self.out_of_bounds_options_values[self.out_of_bounds_handling.get()], 'trials', 'out_of_bounds_handling')
            cfg_set(self.baseline_norm_options_values[self.baseline_norm.get()], 'trials', 'baseline_norm')
            cfg_set((self.baseline_epoch_start.get(), self.baseline_epoch_end.get()), 'trials', 'baseline_epoch')
            cfg_set(self.concat_bidir_stimpairs.get(), 'trials', 'concat_bidirectional_pairs')
            cfg_set(self.min_stimpair_trials.get(), 'trials', 'minimum_stimpair_trials')
            cfg_set(lst_measured_channel_types, 'channels', 'measured_types')
            cfg_set(lst_stim_channel_types, 'channels', 'stim_types')

            #
            self.root.grab_release()
            self.root.destroy()

        def cancel(self):
            self.root.grab_release()
            self.root.destroy()

        def defaults(self):
            config_defaults = create_default_config()
            self.trial_epoch_start.set(config_defaults['trials']['trial_epoch'][0])
            self.trial_epoch_end.set(config_defaults['trials']['trial_epoch'][1])
            self.out_of_bounds_handling.set(self.out_of_bounds_options[config_defaults['trials']['out_of_bounds_handling']])
            self.baseline_norm.set(self.baseline_norm_options[config_defaults['trials']['baseline_norm']])
            self.baseline_epoch_start.set(config_defaults['trials']['baseline_epoch'][0])
            self.baseline_epoch_end.set(config_defaults['trials']['baseline_epoch'][1])
            self.concat_bidir_stimpairs.set(config_defaults['trials']['concat_bidirectional_pairs'])
            self.min_stimpair_trials.set(config_defaults['trials']['minimum_stimpair_trials'])

            measured_channel_types = [x.lower() for x in config_defaults['channels']['measured_types']]
            _btn_set_state(self.btn_measured_channel_types_ecog, False, 'ecog' in measured_channel_types)
            _btn_set_state(self.btn_measured_channel_types_seeg, False, 'seeg' in measured_channel_types)
            _btn_set_state(self.btn_measured_channel_types_dbs, False, 'dbs' in measured_channel_types)
            stim_channel_types = [x.lower() for x in config_defaults['channels']['stim_types']]
            _btn_set_state(self.btn_stim_channel_types_ecog, False, 'ecog' in stim_channel_types)
            _btn_set_state(self.btn_stim_channel_types_seeg, False, 'seeg' in stim_channel_types)
            _btn_set_state(self.btn_stim_channel_types_dbs, False, 'dbs' in stim_channel_types)


    #
    # metrics & detection configuration dialog
    #
    class MetricsAndDetectionDialog(object):

        detect_method_options = {'std_base': 'Deviation from baseline', 'cross_proj': 'Cross-projection metric', 'waveform': 'Waveform metric'}
        detect_method_options_values = {v: k for k, v in detect_method_options.items()}

        def _update_metrics_cross_proj_controls(self):
            new_state = 'normal' if self.metrics_cross_proj_enabled.get() else 'disabled'
            self.lbl_metrics_cross_proj_epoch.configure(state=new_state)
            self.txt_metrics_cross_proj_epoch_start.configure(state=new_state)
            self.lbl_metrics_cross_proj_epoch_range.configure(state=new_state)
            self.txt_metrics_cross_proj_epoch_end.configure(state=new_state)

        def _update_metrics_waveform_controls(self):
            new_state = 'normal' if self.metrics_waveform_enabled.get() else 'disabled'
            self.lbl_metrics_waveform_epoch.configure(state=new_state)
            self.txt_metrics_waveform_epoch_start.configure(state=new_state)
            self.lbl_metrics_waveform_epoch_range.configure(state=new_state)
            self.txt_metrics_waveform_epoch_end.configure(state=new_state)
            self.lbl_metrics_waveform_bandpass.configure(state=new_state)
            self.txt_metrics_waveform_bandpass_start.configure(state=new_state)
            self.lbl_metrics_waveform_bandpass_range.configure(state=new_state)
            self.txt_metrics_waveform_bandpass_end.configure(state=new_state)

        def __init__(self, parent):
            pd_window_height = 600
            pd_window_width = 630

            # retrieve values from config
            self.metrics_cross_proj_enabled = tk.IntVar(value=cfg('metrics', 'cross_proj', 'enabled'))
            self.metrics_cross_proj_epoch_start = tk.DoubleVar(value=cfg('metrics', 'cross_proj', 'epoch')[0])
            self.metrics_cross_proj_epoch_end = tk.DoubleVar(value=cfg('metrics', 'cross_proj', 'epoch')[1])
            self.metrics_waveform_enabled = tk.IntVar(value=cfg('metrics', 'waveform', 'enabled'))
            self.metrics_waveform_epoch_start = tk.DoubleVar(value=cfg('metrics', 'waveform', 'epoch')[0])
            self.metrics_waveform_epoch_end = tk.DoubleVar(value=cfg('metrics', 'waveform', 'epoch')[1])
            self.metrics_waveform_bandpass_start = tk.IntVar(value=cfg('metrics', 'waveform', 'bandpass')[0])
            self.metrics_waveform_bandpass_end = tk.IntVar(value=cfg('metrics', 'waveform', 'bandpass')[1])

            self.detect_neg = tk.IntVar(value=cfg('detection', 'negative'))
            self.detect_pos = tk.IntVar(value=cfg('detection', 'positive'))
            self.detect_peak_search_epoch_start = tk.DoubleVar(value=cfg('detection', 'peak_search_epoch')[0])
            self.detect_peak_search_epoch_end = tk.DoubleVar(value=cfg('detection', 'peak_search_epoch')[1])
            self.detect_response_search_epoch_start = tk.DoubleVar(value=cfg('detection', 'response_search_epoch')[0])
            self.detect_response_search_epoch_end = tk.DoubleVar(value=cfg('detection', 'response_search_epoch')[1])
            self.detect_method = tk.StringVar(value=self.detect_method_options[str(cfg('detection', 'method'))])

            #
            # elements
            #
            self.root = tk.Toplevel(parent)
            self.root.title('Metrics and Detection')
            self.root.geometry("{}x{}+{}+{}".format(pd_window_width, pd_window_height,
                                      int((win.winfo_screenwidth() / 2) - (pd_window_width / 2)),
                                      int((win.winfo_screenheight() / 2) - (pd_window_height / 2))))
            self.root.resizable(False, False)

            #
            pd_y_pos = 15
            self.chk_metrics_cross_proj = tk.Checkbutton(self.root, text='Cross projection metric:', anchor="w", variable=self.metrics_cross_proj_enabled, onvalue=1, offvalue=0, command=self._update_metrics_cross_proj_controls)
            self.chk_metrics_cross_proj.place(x=10, y=pd_y_pos, width=pd_window_width, height=30)
            pd_y_pos += 32
            metrics_cross_proj_state = 'normal' if self.metrics_cross_proj_enabled.get() else 'disabled'
            self.lbl_metrics_cross_proj_epoch = tk.Label(self.root, text="Window", anchor='e', state=metrics_cross_proj_state)
            self.lbl_metrics_cross_proj_epoch.place(x=5, y=pd_y_pos + 2, width=245, height=20)
            self.txt_metrics_cross_proj_epoch_start = ttk.Entry(self.root, textvariable=self.metrics_cross_proj_epoch_start, state=metrics_cross_proj_state, justify='center', validate = 'key', validatecommand=(self.root.register(_txt_double_input_validate), '%S', '%P'))
            self.txt_metrics_cross_proj_epoch_start.place(x=260, y=pd_y_pos, width=70, height=25)
            self.lbl_metrics_cross_proj_epoch_range = tk.Label(self.root, text="-", state=metrics_cross_proj_state)
            self.lbl_metrics_cross_proj_epoch_range.place(x=335, y=pd_y_pos, width=30, height=25)
            self.txt_metrics_cross_proj_epoch_end = ttk.Entry(self.root, textvariable=self.metrics_cross_proj_epoch_end, state=metrics_cross_proj_state, justify='center', validate = 'key', validatecommand=(self.root.register(_txt_double_input_validate), '%S', '%P'))
            self.txt_metrics_cross_proj_epoch_end.place(x=370, y=pd_y_pos, width=70, height=25)
            pd_y_pos += 42
            self.chk_metrics_waveform = tk.Checkbutton(self.root, text='Waveform metric:', anchor="w", variable=self.metrics_waveform_enabled, onvalue=1, offvalue=0, command=self._update_metrics_waveform_controls)
            self.chk_metrics_waveform.place(x=10, y=pd_y_pos, width=pd_window_width, height=30)
            pd_y_pos += 32
            metrics_waveform_state = 'normal' if self.metrics_waveform_enabled.get() else 'disabled'
            self.lbl_metrics_waveform_epoch = tk.Label(self.root, text="Window", anchor='e', state=metrics_waveform_state)
            self.lbl_metrics_waveform_epoch.place(x=5, y=pd_y_pos + 2, width=245, height=20)
            self.txt_metrics_waveform_epoch_start = ttk.Entry(self.root, textvariable=self.metrics_waveform_epoch_start, state=metrics_waveform_state, justify='center', validate = 'key', validatecommand=(self.root.register(_txt_double_input_validate), '%S', '%P'))
            self.txt_metrics_waveform_epoch_start.place(x=260, y=pd_y_pos, width=70, height=25)
            self.lbl_metrics_waveform_epoch_range = tk.Label(self.root, text="-", state=metrics_waveform_state)
            self.lbl_metrics_waveform_epoch_range.place(x=335, y=pd_y_pos, width=30, height=25)
            self.txt_metrics_waveform_epoch_end = ttk.Entry(self.root, textvariable=self.metrics_waveform_epoch_end, state=metrics_waveform_state, justify='center', validate = 'key', validatecommand=(self.root.register(_txt_double_input_validate), '%S', '%P'))
            self.txt_metrics_waveform_epoch_end.place(x=370, y=pd_y_pos, width=70, height=25)
            pd_y_pos += 32
            self.lbl_metrics_waveform_bandpass = tk.Label(self.root, text="Bandpass frequency range", anchor='e', state=metrics_waveform_state)
            self.lbl_metrics_waveform_bandpass.place(x=5, y=pd_y_pos + 2, width=245, height=20)
            self.txt_metrics_waveform_bandpass_start = ttk.Entry(self.root, textvariable=self.metrics_waveform_bandpass_start, state=metrics_waveform_state, justify='center', validate = 'key', validatecommand=(self.root.register(_txt_integer_pos_input_validate), '%S', '%P'))
            self.txt_metrics_waveform_bandpass_start.place(x=260, y=pd_y_pos, width=70, height=25)
            self.lbl_metrics_waveform_bandpass_range = tk.Label(self.root, text="-", state=metrics_waveform_state)
            self.lbl_metrics_waveform_bandpass_range.place(x=335, y=pd_y_pos, width=30, height=25)
            self.txt_metrics_waveform_bandpass_end = ttk.Entry(self.root, textvariable=self.metrics_waveform_bandpass_end, state=metrics_waveform_state, justify='center', validate = 'key', validatecommand=(self.root.register(_txt_integer_pos_input_validate), '%S', '%P'))
            self.txt_metrics_waveform_bandpass_end.place(x=370, y=pd_y_pos, width=70, height=25)
            pd_y_pos += 80

            tk.Label(self.root, text="Response Detection:", anchor='w').place(x=10, y=pd_y_pos + 2, width=245, height=20)
            pd_y_pos += 36
            tk.Label(self.root, text="Negative deflections", anchor='e').place(x=5, y=pd_y_pos + 2, width=245, height=20)
            tk.Checkbutton(self.root, text='', anchor="w", variable=self.detect_neg, onvalue=1, offvalue=0).place(x=256, y=pd_y_pos, width=pd_window_width, height=30)
            pd_y_pos += 32

            tk.Label(self.root, text="Positive deflections", anchor='e').place(x=5, y=pd_y_pos + 2, width=245, height=20)
            tk.Checkbutton(self.root, text='', anchor="w", variable=self.detect_pos, onvalue=1, offvalue=0).place(x=256, y=pd_y_pos, width=pd_window_width, height=30)
            pd_y_pos += 38
            tk.Label(self.root, text="Peak search window", anchor='e').place(x=5, y=pd_y_pos + 2, width=245, height=20)
            self.txt_detect_peak_search_epoch_start = ttk.Entry(self.root, textvariable=self.detect_peak_search_epoch_start, justify='center', validate = 'key', validatecommand=(self.root.register(_txt_double_input_validate), '%S', '%P'))
            self.txt_detect_peak_search_epoch_start.place(x=260, y=pd_y_pos, width=70, height=25)
            tk.Label(self.root, text="-").place(x=335, y=pd_y_pos, width=30, height=25)
            self.txt_detect_peak_search_epoch_end = ttk.Entry(self.root, textvariable=self.detect_peak_search_epoch_end, justify='center', validate = 'key', validatecommand=(self.root.register(_txt_double_input_validate), '%S', '%P'))
            self.txt_detect_peak_search_epoch_end.place(x=370, y=pd_y_pos, width=70, height=25)
            pd_y_pos += 32
            tk.Label(self.root, text="Response search window", anchor='e').place(x=5, y=pd_y_pos + 2, width=245, height=20)
            self.txt_detect_response_search_epoch_start = ttk.Entry(self.root, textvariable=self.detect_response_search_epoch_start, justify='center', validate = 'key', validatecommand=(self.root.register(_txt_double_input_validate), '%S', '%P'))
            self.txt_detect_response_search_epoch_start.place(x=260, y=pd_y_pos, width=70, height=25)
            tk.Label(self.root, text="-").place(x=335, y=pd_y_pos, width=30, height=25)
            self.txt_detect_response_search_epoch_end = ttk.Entry(self.root, textvariable=self.detect_response_search_epoch_end, justify='center', validate = 'key', validatecommand=(self.root.register(_txt_double_input_validate), '%S', '%P'))
            self.txt_detect_response_search_epoch_end.place(x=370, y=pd_y_pos, width=70, height=25)
            pd_y_pos += 42
            self.lbl_detect_method_handling = tk.Label(self.root, text="Method", anchor='e', state='disabled')
            self.lbl_detect_method_handling.place(x=5, y=pd_y_pos + 2, width=245, height=20)
            self.cmb_detect_method_handling = ttk.Combobox(self.root, textvariable=self.detect_method, values=list(self.detect_method_options_values.keys()), state='disabled')
            self.cmb_detect_method_handling.bind("<Key>", lambda e: "break")
            self.cmb_detect_method_handling.bind("<<ComboboxSelected>>", _update_combo_losefocus)
            self.cmb_detect_method_handling.bind("<FocusIn>", _update_combo_losefocus)
            self.cmb_detect_method_handling.place(x=260, y=pd_y_pos, width=300, height=25)
            #pd_y_pos += 32
            #
            #self.lbl_detect_stdbase_epoch = tk.Label(self.root, text="Baseline window", anchor='e').place(x=5, y=pd_y_pos + 2, width=245, height=20)
            #self.txt_detect_stdbase_epoch_start = ttk.Entry(self.root, textvariable=self.detect_response_search_epoch_start, justify='center', validate = 'key', validatecommand=(self.root.register(_txt_double_input_validate), '%S', '%P'))
            #self.txt_detect_stdbase_epoch_start.place(x=260, y=pd_y_pos, width=70, height=25)
            #self.lbl_detect_stdbase_epoch_range = tk.Label(self.root, text="-").place(x=335, y=pd_y_pos, width=30, height=25)
            #self.txt_detect_stdbase_epoch_end = ttk.Entry(self.root, textvariable=self.detect_response_search_epoch_end, justify='center', validate = 'key', validatecommand=(self.root.register(_txt_double_input_validate), '%S', '%P'))
            #self.txt_detect_stdbase_epoch_end.place(x=370, y=pd_y_pos, width=70, height=25)

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

            # check input values
            if self.metrics_cross_proj_enabled.get():
                if self.metrics_cross_proj_epoch_end.get() < self.metrics_cross_proj_epoch_start.get():
                    messagebox.showerror(title='Invalid input', message='Invalid window (start and end values) for the cross projection metric. '
                                                                        'The given end-point (' + str(self.metrics_cross_proj_epoch_end.get()) + ') lies before the start-point (' + str(self.metrics_cross_proj_epoch_start.get()) + ')')
                    self.txt_metrics_cross_proj_epoch_start.focus()
                    self.txt_metrics_cross_proj_epoch_start.select_range(0, tk.END)
                    return
            if self.metrics_waveform_enabled.get():
                if self.metrics_waveform_epoch_end.get() < self.metrics_waveform_epoch_start.get():
                    messagebox.showerror(title='Invalid input', message='Invalid window (start and end values) for the waveform metric. '
                                                                        'The given end-point (' + str(self.metrics_waveform_epoch_end.get()) + ') lies before the start-point (' + str(self.metrics_waveform_epoch_start.get()) + ')')
                    self.txt_metrics_waveform_epoch_start.focus()
                    self.txt_metrics_waveform_epoch_start.select_range(0, tk.END)
                    return
                if self.metrics_waveform_bandpass_start.get() < 1:
                    messagebox.showerror(title='Invalid input', message='Invalid lower cutoff frequency for the waveform bandpass metric, the value should be 1 or higher')
                    self.txt_metrics_waveform_bandpass_start.focus()
                    self.txt_metrics_waveform_bandpass_start.select_range(0, tk.END)
                    return
                if self.metrics_waveform_bandpass_end.get() < self.metrics_waveform_bandpass_start.get():
                    messagebox.showerror(title='Invalid input', message='Invalid frequency range (start and end values) for the waveform bandpass metric. '
                                                                        'The end of the range (' + str(self.metrics_waveform_bandpass_end.get()) + ') lies before the start of the range (' + str(self.metrics_waveform_bandpass_start.get()) + ')')
                    self.txt_metrics_waveform_bandpass_start.focus()
                    self.txt_metrics_waveform_bandpass_start.select_range(0, tk.END)
                    return
            if self.detect_peak_search_epoch_end.get() <= self.detect_peak_search_epoch_start.get():
                messagebox.showerror(title='Invalid input', message='Invalid window (start and end values) for the peak search epoch setting. '
                                                                    'The given end-point (' + str(self.detect_peak_search_epoch_end.get()) + ') lies on or before the start-point (' + str(self.detect_peak_search_epoch_start.get()) + ')')
                self.txt_detect_peak_search_epoch_start.focus()
                self.txt_detect_peak_search_epoch_start.select_range(0, tk.END)
                return
            if self.detect_response_search_epoch_end.get() <= self.detect_response_search_epoch_start.get():
                messagebox.showerror(title='Invalid input', message='Invalid window (start and end values) for the response search epoch setting. '
                                                                    'The given end-point (' + str(self.detect_response_search_epoch_end.get()) + ') lies on or before the start-point (' + str(self.detect_response_search_epoch_start.get()) + ')')
                self.txt_detect_response_search_epoch_start.focus()
                self.txt_detect_response_search_epoch_start.select_range(0, tk.END)
                return
            # TODO: check inside of trial epoch

            # update config
            cfg_set(self.metrics_cross_proj_enabled.get(), 'metrics', 'cross_proj', 'enabled')
            if self.metrics_cross_proj_enabled.get():
                cfg_set((self.metrics_cross_proj_epoch_start.get(), self.metrics_cross_proj_epoch_end.get()), 'metrics', 'cross_proj', 'epoch')
            cfg_set(self.metrics_waveform_enabled.get(), 'metrics', 'waveform', 'enabled')
            if self.metrics_waveform_enabled.get():
                cfg_set((self.metrics_waveform_epoch_start.get(), self.metrics_waveform_epoch_end.get()), 'metrics', 'waveform', 'epoch')
                cfg_set((self.metrics_waveform_bandpass_start.get(), self.metrics_waveform_bandpass_end.get()), 'metrics', 'waveform', 'bandpass')
            cfg_set(self.detect_neg.get(), 'detection', 'negative')
            cfg_set(self.detect_pos.get(), 'detection', 'positive')
            cfg_set((self.detect_peak_search_epoch_start.get(), self.detect_peak_search_epoch_end.get()), 'detection', 'peak_search_epoch')
            cfg_set((self.detect_response_search_epoch_start.get(), self.detect_response_search_epoch_end.get()), 'detection', 'response_search_epoch')

            #
            self.root.grab_release()
            self.root.destroy()

        def cancel(self):
            self.root.grab_release()
            self.root.destroy()

        def defaults(self):
            config_defaults = create_default_config()
            self.metrics_cross_proj_enabled.set(config_defaults['metrics']['cross_proj']['enabled'])
            self.metrics_cross_proj_epoch_start.set(config_defaults['metrics']['cross_proj']['epoch'][0])
            self.metrics_cross_proj_epoch_end.set(config_defaults['metrics']['cross_proj']['epoch'][1])
            self.metrics_waveform_enabled.set(config_defaults['metrics']['waveform']['enabled'])
            self.metrics_waveform_epoch_start.set(config_defaults['metrics']['waveform']['epoch'][0])
            self.metrics_waveform_epoch_end.set(config_defaults['metrics']['waveform']['epoch'][1])
            self.metrics_waveform_bandpass_start.set(config_defaults['metrics']['waveform']['bandpass'][0])
            self.metrics_waveform_bandpass_end.set(config_defaults['metrics']['waveform']['bandpass'][1])
            self.detect_neg.set(config_defaults['detection']['negative'])
            self.detect_pos.set(config_defaults['detection']['positive'])
            self.detect_peak_search_epoch_start.set(config_defaults['detection']['peak_search_epoch'][0])
            self.detect_peak_search_epoch_end.set(config_defaults['detection']['peak_search_epoch'][1])
            self.detect_response_search_epoch_start.set(config_defaults['detection']['response_search_epoch'][0])
            self.detect_response_search_epoch_end.set(config_defaults['detection']['response_search_epoch'][1])

            self._update_metrics_cross_proj_controls()
            self._update_metrics_waveform_controls()


    #
    # visualization configuration dialog
    #
    class VisualizationDialog(object):

        def __init__(self, parent):
            pd_window_height = 350
            pd_window_width = 460

            # retrieve values from config
            self.visualize_neg = tk.IntVar(value=cfg('visualization', 'negative'))
            self.visualize_pos = tk.IntVar(value=cfg('visualization', 'positive'))
            self.visualize_x_axis_epoch_start = tk.DoubleVar(value=cfg('visualization', 'x_axis_epoch')[0])
            self.visualize_x_axis_epoch_end = tk.DoubleVar(value=cfg('visualization', 'x_axis_epoch')[1])
            self.visualize_blank_stim_epoch_start = tk.DoubleVar(value=cfg('visualization', 'blank_stim_epoch')[0])
            self.visualize_blank_stim_epoch_end = tk.DoubleVar(value=cfg('visualization', 'blank_stim_epoch')[1])
            self.generate_electrode_images = tk.IntVar(value=cfg('visualization', 'generate_electrode_images'))
            self.generate_stimpair_images = tk.IntVar(value=cfg('visualization', 'generate_stimpair_images'))
            self.generate_matrix_images = tk.IntVar(value=cfg('visualization', 'generate_matrix_images'))


            #
            # elements
            #
            self.root = tk.Toplevel(parent)
            self.root.title('Visualization')
            self.root.geometry("{}x{}+{}+{}".format(pd_window_width, pd_window_height,
                                      int((win.winfo_screenwidth() / 2) - (pd_window_width / 2)),
                                      int((win.winfo_screenheight() / 2) - (pd_window_height / 2))))
            self.root.resizable(False, False)

            #
            pd_y_pos = 15
            tk.Label(self.root, text="Negative deflections", anchor='e').place(x=5, y=pd_y_pos + 2, width=225, height=20)
            tk.Checkbutton(self.root, text='', anchor="w", variable=self.visualize_neg, onvalue=1, offvalue=0).place(x=236, y=pd_y_pos, width=pd_window_width, height=30)
            pd_y_pos += 30
            tk.Label(self.root, text="Positive deflections", anchor='e').place(x=5, y=pd_y_pos + 2, width=225, height=20)
            tk.Checkbutton(self.root, text='', anchor="w", variable=self.visualize_pos, onvalue=1, offvalue=0).place(x=236, y=pd_y_pos, width=pd_window_width, height=30)
            pd_y_pos += 42

            tk.Label(self.root, text="X-axis window", anchor='e').place(x=5, y=pd_y_pos + 2, width=225, height=20)
            self.txt_visualize_x_axis_epoch_start = ttk.Entry(self.root, textvariable=self.visualize_x_axis_epoch_start, justify='center', validate = 'key', validatecommand=(self.root.register(_txt_double_input_validate), '%S', '%P'))
            self.txt_visualize_x_axis_epoch_start.place(x=240, y=pd_y_pos, width=70, height=25)
            tk.Label(self.root, text="-").place(x=315, y=pd_y_pos, width=30, height=25)
            self.txt_visualize_x_axis_epoch_end = ttk.Entry(self.root, textvariable=self.visualize_x_axis_epoch_end, justify='center', validate = 'key', validatecommand=(self.root.register(_txt_double_input_validate), '%S', '%P'))
            self.txt_visualize_x_axis_epoch_end.place(x=350, y=pd_y_pos, width=70, height=25)
            pd_y_pos += 32
            tk.Label(self.root, text="Blank stimulus window", anchor='e').place(x=5, y=pd_y_pos + 2, width=225, height=20)
            self.txt_visualize_blank_stim_epoch_start = ttk.Entry(self.root, textvariable=self.visualize_blank_stim_epoch_start, justify='center', validate = 'key', validatecommand=(self.root.register(_txt_double_input_validate), '%S', '%P'))
            self.txt_visualize_blank_stim_epoch_start.place(x=240, y=pd_y_pos, width=70, height=25)
            tk.Label(self.root, text="-").place(x=315, y=pd_y_pos, width=30, height=25)
            self.txt_visualize_blank_stim_epoch_end = ttk.Entry(self.root, textvariable=self.visualize_blank_stim_epoch_end, justify='center', validate = 'key', validatecommand=(self.root.register(_txt_double_input_validate), '%S', '%P'))
            self.txt_visualize_blank_stim_epoch_end.place(x=350, y=pd_y_pos, width=70, height=25)
            pd_y_pos += 42

            tk.Label(self.root, text="Generate electrode images", anchor='e').place(x=5, y=pd_y_pos + 2, width=225, height=20)
            tk.Checkbutton(self.root, text='', anchor="w", variable=self.generate_electrode_images, onvalue=1, offvalue=0).place(x=236, y=pd_y_pos, width=pd_window_width, height=30)
            pd_y_pos += 30
            tk.Label(self.root, text="Generate stimulus-pair images", anchor='e').place(x=5, y=pd_y_pos + 2, width=225, height=20)
            tk.Checkbutton(self.root, text='', anchor="w", variable=self.generate_stimpair_images, onvalue=1, offvalue=0).place(x=236, y=pd_y_pos, width=pd_window_width, height=30)
            pd_y_pos += 30
            tk.Label(self.root, text="Generate matrix images", anchor='e').place(x=5, y=pd_y_pos + 2, width=225, height=20)
            tk.Checkbutton(self.root, text='', anchor="w", variable=self.generate_matrix_images, onvalue=1, offvalue=0).place(x=236, y=pd_y_pos, width=pd_window_width, height=30)


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

            # check input values
            if self.visualize_x_axis_epoch_end.get() <= self.visualize_x_axis_epoch_start.get():
                messagebox.showerror(title='Invalid input', message='Invalid window (start and end values) for the x-axis epoch setting. '
                                                                    'The given end-point (' + str(self.visualize_x_axis_epoch_end.get()) + ') lies on or before the start-point (' + str(self.visualize_x_axis_epoch_start.get()) + ')')
                self.txt_visualize_x_axis_epoch_start.focus()
                self.txt_visualize_x_axis_epoch_start.select_range(0, tk.END)
                return
            if self.visualize_blank_stim_epoch_end.get() <= self.visualize_blank_stim_epoch_start.get():
                messagebox.showerror(title='Invalid input', message='Invalid window (start and end values) for the blank stimulus epoch setting. '
                                                                    'The given end-point (' + str(self.visualize_blank_stim_epoch_end.get()) + ') lies on or before the start-point (' + str(self.visualize_blank_stim_epoch_start.get()) + ')')
                self.txt_visualize_blank_stim_epoch_start.focus()
                self.txt_visualize_blank_stim_epoch_start.select_range(0, tk.END)
                return
            # TODO: check inside of trial epoch

            # update config
            cfg_set(self.visualize_neg.get(), 'visualization', 'negative')
            cfg_set(self.visualize_pos.get(), 'visualization', 'positive')
            cfg_set((self.visualize_x_axis_epoch_start.get(), self.visualize_x_axis_epoch_end.get()), 'visualization', 'x_axis_epoch')
            cfg_set((self.visualize_blank_stim_epoch_start.get(), self.visualize_blank_stim_epoch_end.get()), 'visualization', 'blank_stim_epoch')
            cfg_set(self.generate_electrode_images.get(), 'visualization', 'generate_electrode_images')
            cfg_set(self.generate_stimpair_images.get(), 'visualization', 'generate_stimpair_images')
            cfg_set(self.generate_matrix_images.get(), 'visualization', 'generate_matrix_images')

            #
            self.root.grab_release()
            self.root.destroy()

        def cancel(self):
            self.root.grab_release()
            self.root.destroy()

        def defaults(self):
            config_defaults = create_default_config()
            self.visualize_neg.set(config_defaults['visualization']['negative'])
            self.visualize_pos.set(config_defaults['visualization']['positive'])
            self.visualize_x_axis_epoch_start.set(config_defaults['visualization']['x_axis_epoch'][0])
            self.visualize_x_axis_epoch_end.set(config_defaults['visualization']['x_axis_epoch'][1])
            self.visualize_blank_stim_epoch_start.set(config_defaults['visualization']['blank_stim_epoch'][0])
            self.visualize_blank_stim_epoch_end.set(config_defaults['visualization']['blank_stim_epoch'][1])
            self.generate_electrode_images.set(config_defaults['visualization']['generate_electrode_images'])
            self.generate_stimpair_images.set(config_defaults['visualization']['generate_stimpair_images'])
            self.generate_matrix_images.set(config_defaults['visualization']['generate_matrix_images'])


    #
    # main window
    #
    window_width = 640
    window_height = 800

    # open window
    win = tk.Tk()
    icon_64 = 'iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAIM0lEQVR4nO3be4xdVRUG8N+eNM2ENKRBU6tBovKIgdYoYtFiBeUhigoFKcKxRZ4RlSDWB1CJIYgNMRW0KKBAEXugoFYeFRCqECiCiKTW0iCp' \
              'WEttyKTB2kxIM2nm+MfetzO9rznnzrS96nzJzdxzzz5r77322mt9a+0zjGMc4/h/RtjbA6ihyL0D70QvNmJdyLy+u/vdLQoocj04Bh/Dodgv3dqIp7A8ZDalSV+COZhaJ2YHVmEJloXMwO4Y65groMidimvE1WyFQTyLIzChhNgXcXbIPDv6Ee6K' \
              'MVNAkdsHN2LeWMmsw3bci/1Fi+rHS3gS94ZMXydCx0QBRa4XD4lmvzcwgDuwoKoiOlZAkdtf3N9TMBundiprDLEZnwyZ58s+UEkBRW4yvoCzcUi1se0xbMWxZZVQWgFF7ix8H2/scGCtsE3cz28ZQ5mb8N4y26GnjLQidw1yYz/5AUxPn+1jKHd/' \
              'LC7TcEQFFLnLcMVoR9QCg+gLmdewfoxlzylytxa5g9o1aquAInc4rq7Q6e04WfnJTBSZHzGklcEKPFyy7bl4ochdV+QmNWswkgUsVI6o1HB3yNyPI7FUXOFW6MM5IbM1XT9UQv5SUcGzKc0MJ+LLeKrIG/1MSydY5N6GvynpJxI246MhszbJOBRn' \
              'ioyvRnU34FEsDZltw/qbnPrbT3NswPSQ6S9y5+MnFcZVwxocFTL9tR+aKiBx+UWi5qqiHwtw00j8PeUCl0sEpsjNw09bNM9C5s4iNw1/NLR1quKmkLmodtFg3kXuENysc1Y3SQyX/bityO2bfq9ldlMwE6fjFNFE9xEneEeROxjfrJO5Cb9I3xfp' \
              'fPJwYZG7MWTWUGfeRe5T+JOxobQ1InIV/pU+/8Y/8XMxA5yY2pxV5C6EkLkSl7JLKrw8ZAbS6h83ynH1YP7wC1DkTkgDa+otE9bj6CRgY5t2a7EmbaUTUz+TxJVuhcVF7hMQMtfjYFyJZ/BYanOcaj6pFU4p8mj9AYrcVPzFyERneciclp7pxVfx' \
              'LY1b6cyQWVbkZuAPFQZ2Qcjc0upmkdtPtM7j8RlMriC7HoeFzLqaApbgcyUeGsQyLBzm6T8uWk5tde/E3JAZLHIPEFe1BJ7Dkem5RSIzfACrQ9bIElPUuBxfUS1U13B0yDwRUmz8R0UhO3ADvhYyO4rccWmwD4rObHvKHfIKMmeHzL1F7hhDJk9U' \
              'xEbREb4sRoD7Q+ZVKHIfxC9F59oM60V/NKfu9+NDZmVPulFVgxPEEHl3kZsYMivxAZyeJv8h1eL0VlF5RAc4HL1i5vkRnC9GqL8ndrdPyKzCh7Glidx78J6QOUPkAPV9moBZFQbaj3UiKXkVr4j7sC9kVkOR+2waZDuHV4/nkpefLDrNkdArLsCM' \
              'IndSyKwrcnPxa9FJrsdVIbM0jelEsXYxHJuICmibLCT8DtdhZbP9mDp5l1gLHGnPPy/y+RfwmridNqd7hxoKjWUwU7TCk0Lm4SJ3sriyvw9ZpOGJNS7WaOXn4juhyP1V6+LGFlwUsp0kREoqDhLz96k4UDTPGdqHqA2ic1zVqkHyJY+2kdEKF4fM' \
              'DXWyZogL0oo3bMNbJ7AzGanHVswKmRdTPP80voj3q7ZKNTwcMqtS/D1CXL0DRaX9KmQeobPCJi4ucj9KEWQOvoHDR3hmX8ybIKahM5o0WJcmPwU/wwkdDq6GealyPFPjtpuGR9JYtqXBVcEhScYaHGXkyddweg9+2+LmzCL3mLhXRzt5olOcp7nP' \
              'OaLITU7+5cEm98tgZvr7SoVnDujBcobS0jocY+zLYM3QayhOf1d0jM3QL/KP03AY3pQ+0w0VSSqdINWY4NUaM7A9jfVivr+9yC3EZU3azA+Z77UTUuS+jmtL9rmhFhoWiqnptLKj3Q04SEx+Foj5xVk4oK7NrCL3HN6N9xlif5twaSqwvLlKpzsL' \
              'Iql4+KTGQ8o9iUEx03wCv1F++23G2xOZesqQPxgJj+9SEUpKuE8ja+p2XBAyt6R6wZ+VT5nn79IwZNaLpvVthupmLbBD5NrnaeTZzbBB5BEtiVCHuAe3pe/zlZ/8Vtzerig6Wcy5jxX3Ws+wB58Wz+xfTm1r7wPMFmPwVJF6bhGVcx9WpMyxV/T0' \
              'n9dZGlvDoBgR5ocsRo1UzlusXNi+JGR+sNfeEElU9Vqdld/6RLO/P8mahpdD5vW0GJeJ5xmtrOFx8fxwcK+/IpMyteu0f6FiONaIJ8Ab02S/JFpUn5i3rEhyrxBzgWaYVctJ9roC2Fleu1UMfe3wkljX35IKOUvsau6D4pa4Psm9S9zGwzGAN9TO' \
              'BsaiwDhqJAp8tvgGSCvsEGuNW1K0elrjXu/BonS+QHSK9adTjw8/GOkKBUByZOcZqg3U486QeT5Zy30aSVINPfhhKvGfqnGOS+obdw3SKfGCFrdvTn8vNDJPmSQqqf6IfLWhAxZ0mQISlmm0gtfY+YbY3A7l9osFmV0Sra5TQPIHd9T9vDZxiEli' \
              'HlAVr+O0Wil/OLpOAQmP1V2/mv5OUZ08bRBj/iPNbnarAuoLrzWvXaXSvF2k9NND5plWjUZDRfckagopW4tcK4bMBpOvR7cqoL4iVLsuM94VOKPsi9bdqoB68tLuVZvhWCEesbUqqTWgW31Aqwm3W7C14sqXnjzdq4BKk0jt53by/wXdqoCyJl/D' \
              'j2tnk1Xxv6CAAeWrwA3oVgVU2QIPhqzt6zpt0a0KqGIBd42mo/92BQxg5Wg66lYFlN0Cq1MK3TG6lQj1ia/YThQXaV36fT3OSd8HlH/BehzjGMc4muI/NG5lZrDw9tYAAAAASUVORK5CYII='
    app_icon = tk.PhotoImage(height=64, width=64, data=icon_64)
    win.iconphoto(False, app_icon)

    win.title('Evoked Response detection')
    win.geometry("{}x{}+{}+{}".format(window_width, window_height,
                                      int((win.winfo_screenwidth() / 2) - (window_width / 2)),
                                      int((win.winfo_screenheight() / 2) - (window_height / 2))))
    win.resizable(False, False)

    # window variables
    datasets = None
    datasets_filtered_keys = None
    input_directory = tk.StringVar()
    output_directory = tk.StringVar()
    last_browsed_configfile = ''
    subset_items = tk.StringVar()
    subset_filter = tk.StringVar()
    processing_thread = None
    processing_thread_lock = threading.Lock()

    #
    def update_datasets(dataset_directory):
        nonlocal datasets, datasets_filtered_keys

        # reset datasets and filters
        datasets = None
        datasets_filtered_keys = None
        subset_filter.set('')

        # search for datasets
        try:
            datasets_found = list_bids_datasets(  dataset_directory, VALID_FORMAT_EXTENSIONS,
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
            lst_subsets.configure(background=win['background'], state='disabled')
            btn_subsets_all.config(state='disabled')
            btn_subsets_none.config(state='disabled')
            lbl_subsets_filter.config(state='disabled')
            txt_subsets_filter.config(state='disabled')
            btn_process.config(state='disabled', text='Process')

        # initially no selection, so disable process button
        btn_process.config(state='disabled', text='Start')

        #
        txt_input_browse.config(state='readonly')
        txt_output_browse.config(state='readonly')
        btn_output_browse.config(state='normal')

        #
        update_subset_list('')

    def btn_input_browse_onclick():

        initial_dir = input_directory.get()
        if not initial_dir:
            initial_dir = os.path.abspath(os.path.expanduser(os.path.expandvars('~')))

        folder_selected = filedialog.askdirectory(title='Select BIDS root directory', initialdir=initial_dir)
        if folder_selected is not None and folder_selected != '':
            input_directory.set(os.path.abspath(os.path.expanduser(os.path.expandvars(folder_selected))))

            # provide an initial output directory
            output_dir = output_directory.get()
            if not output_dir:
                output_directory.set(os.path.join(input_directory.get(), 'derivatives', 'erdetect_out'))

            # update the dataset list with the selected folder
            update_datasets(folder_selected)

    def btn_output_browse_onclick():

        initial_dir = output_directory.get()
        if not initial_dir:
            initial_dir = os.path.abspath(os.path.expanduser(os.path.expandvars('~')))

        folder_selected = filedialog.askdirectory(title='Select output directory', initialdir=initial_dir)
        if folder_selected is not None and folder_selected != '':
            output_directory.set(os.path.abspath(os.path.expanduser(os.path.expandvars(folder_selected))))

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
            lst_subsets.configure(background='white', state='normal')
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

    def config_trials_and_channel_types_callback():
        dialog = TrialsAndChannelTypesDialog(win)
        win.wait_window(dialog.root)

    def config_metrics_and_detection_callback():
        dialog = MetricsAndDetectionDialog(win)
        win.wait_window(dialog.root)

    def config_visualization_callback():
        dialog = VisualizationDialog(win)
        win.wait_window(dialog.root)

    def btn_process_start_onclick():
        nonlocal processing_thread, processing_thread_lock

        #
        datasets_to_analyze = [(val['label'], key) for key, val in datasets.items() if val['selected']]

        # create a thread to process the datasets
        processing_thread_lock.acquire()
        if processing_thread is None:
            processing_thread = threading.Thread(target=process_thread, args=(datasets_to_analyze, output_directory.get()), daemon=False)
            processing_thread.start()
        else:
            print('Already started')
        processing_thread_lock.release()

        # disable controls
        btn_process.config(state='disabled')

        # TODO: show only sets to process and disable list
        btn_input_browse.config(state='disabled')
        lst_subsets.configure(background=win['background'], state='disabled')
        btn_subsets_all.config(state='disabled')
        btn_subsets_none.config(state='disabled')
        lbl_subsets_filter.config(state='disabled')
        txt_subsets_filter.config(state='disabled')

        btn_cfg_import.config(state='disabled')
        btn_cfg_preproc.config(state='disabled')
        btn_cfg_trials_channel_types.config(state='disabled')
        btn_cfg_detect_metrics.config(state='disabled')
        btn_cfg_visualization.config(state='disabled')

        btn_output_browse.config(state='disabled')


    def process_thread(process_datasets, output_dir):
        nonlocal processing_thread, processing_thread_lock

        #
        preproc_prioritize_speed = True

        #
        print_config(preproc_prioritize_speed)

        # display subject/subset information
        txt_console.insert(tk.END, 'Participant(s) and subset(s) to process:\n')
        for (name, path) in process_datasets:
            txt_console.insert(tk.END, '    - ' + name + '\n')
        txt_console.insert(tk.END, '\n')
        txt_console.insert(tk.END, 'Output directory:\n' + output_dir + '\n')
        txt_console.insert(tk.END, '\n')
        txt_console.see(tk.END)

        # process
        for (val, path) in process_datasets:
            txt_console.insert(tk.END, '------ Processing ' + name + ' -------\n')
            txt_console.insert(tk.END, '\n')
            txt_console.see(tk.END)

            # process
            path = os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
            try:
                process_subset(path, output_dir, preproc_prioritize_speed=preproc_prioritize_speed)
            except RuntimeError:
                txt_console.insert(tk.END, 'Error while processing dataset, stopping...\n')
                enable_controls()
                # TODO: handle when error

        # empty space and end message
        txt_console.insert(tk.END, '\n\n')
        txt_console.insert(tk.END, '-----------      Finished running      -----------')
        txt_console.see(tk.END)

        #
        enable_controls()

        #
        processing_thread_lock.acquire()
        processing_thread = None
        processing_thread_lock.release()


    def btn_import_config_callback():
        nonlocal last_browsed_configfile

        #
        initial_dir = os.path.split(last_browsed_configfile)[0] if last_browsed_configfile else os.path.abspath(os.path.expanduser(os.path.expandvars('~')))
        file_selected = filedialog.askopenfilename(title='Select JSON file to import', initialdir=initial_dir, filetypes=[("JSON files", "*.json")])
        if file_selected is not None and file_selected != '':
            last_browsed_configfile = os.path.abspath(os.path.expanduser(os.path.expandvars(file_selected)))

            #  read the configuration file
            txt_console.insert(tk.END, 'Importing configuration file:\n' + last_browsed_configfile + '\n')
            if load_config(last_browsed_configfile):
                txt_console.insert(tk.END, '> Import successful\n')
            else:
                txt_console.insert(tk.END, '> Import failed!\n')
            txt_console.see(tk.END)


    def enable_controls():

        btn_input_browse.config(state='normal')
        lst_subsets.configure(background='white', state='normal')
        btn_subsets_all.config(state='normal')
        btn_subsets_none.config(state='normal')
        lbl_subsets_filter.config(state='normal')
        txt_subsets_filter.config(state='normal')
        txt_subsets_filter.config(state='normal')

        btn_cfg_import.config(state='normal')
        btn_cfg_preproc.config(state='normal')
        btn_cfg_trials_channel_types.config(state='normal')
        btn_cfg_detect_metrics.config(state='normal')
        btn_cfg_visualization.config(state='normal')

        btn_output_browse.config(state='normal')
        btn_process.config(state='normal')

    def _txt_no_input_onkey(event):
        # TODO: check for mac
        if event.state == 12 and event.keysym == 'c':
            return
        else:
            return "break"


    # elements
    y_pos = 10
    tk.Label(win, text="BIDS input directory:", anchor='w').place(x=5, y=y_pos, width=window_width - 10, height=20)
    y_pos += 20 + 5
    txt_input_browse = tk.Entry(win, textvariable=input_directory, state='disabled')
    txt_input_browse.place(x=10, y=y_pos, width=window_width - 120, height=25)
    btn_input_browse = tk.Button(win, text="Browse...", command=btn_input_browse_onclick)
    btn_input_browse.place(x=window_width - 105, y=y_pos, width=95, height=25)

    y_pos += 37 + 5
    tk.Label(win, text="Subjects/subsets (click/highlight to include for processing):", anchor='w').place(x=5, y=y_pos, width=window_width - 10, height=20)
    y_pos += 20 + 5
    lst_subsets = tk.Listbox(win, listvariable=subset_items, selectmode="multiple", activestyle=tk.NONE, exportselection=False, state='disabled', background=win['background'])
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
    txt_subsets_filter.place(x=240, y=y_pos, width=window_width - 272, height=25)
    txt_subsets_filter.bind('<KeyRelease>', txt_subsets_filter_onkeyrelease)

    y_pos += 40
    tk.Label(win, text="Configuration:", anchor='w').place(x=5, y=y_pos, width=window_width - 10, height=20)
    y_pos += 20 + 5
    btn_cfg_import = tk.Button(win, text="Import from JSON file...", command=btn_import_config_callback)
    btn_cfg_import.place(x=10, y=y_pos, width=window_width - 20, height=26)
    y_pos += 30 + 5
    config_btn_width = (window_width - 10 - 10 - 10) / 2
    btn_cfg_preproc = tk.Button(win, text="Preprocessing", command=config_preprocessing_callback)
    btn_cfg_preproc.place(x=10, y=y_pos, width=config_btn_width, height=28)
    btn_cfg_trials_channel_types = tk.Button(win, text="Trials and channel-types", command=config_trials_and_channel_types_callback)
    btn_cfg_trials_channel_types.place(x=10 + config_btn_width + 10, y=y_pos, width=config_btn_width, height=28)
    y_pos += 28
    btn_cfg_detect_metrics = tk.Button(win, text="Metrics & Detection", command=config_metrics_and_detection_callback)
    btn_cfg_detect_metrics.place(x=10, y=y_pos, width=config_btn_width, height=28)
    btn_cfg_visualization = tk.Button(win, text="Visualization", command=config_visualization_callback)
    btn_cfg_visualization.place(x=10 + config_btn_width + 10, y=y_pos, width=config_btn_width, height=28)
    # TODO: speed vs memory processing

    y_pos += 45 + 2
    tk.Label(win, text="BIDS output directory:", anchor='w').place(x=5, y=y_pos, width=window_width - 10, height=20)
    y_pos += 20 + 5
    txt_output_browse = tk.Entry(win, textvariable=output_directory, state='disabled')
    txt_output_browse.place(x=10, y=y_pos, width=window_width - 120, height=25)
    btn_output_browse = tk.Button(win, text="Browse...", command=btn_output_browse_onclick, state='disabled')
    btn_output_browse.place(x=window_width - 105, y=y_pos, width=95, height=25)

    y_pos += 45 + 2
    tk.Label(win, text="Process:", anchor='w').place(x=5, y=y_pos, width=window_width - 10, height=20)
    y_pos += 20 + 2
    btn_process = tk.Button(win, text="Start", command=btn_process_start_onclick, state='disabled')
    btn_process.place(x=10, y=y_pos, width=window_width - 20, height=40)
    y_pos += 40 + 2
    txt_console = tk.Text(win, highlightthickness = 0, borderwidth=1, relief="solid", undo=False, maxundo=-1, background=win['background'])
    txt_console.bind("<Key>", lambda e: _txt_no_input_onkey(e))

    scr_subsets = tk.Scrollbar(win, orient='vertical')
    txt_console.place(x=12, y=y_pos, width=window_width - 45, height=120)
    scr_subsets.place(x=window_width - 33, y=y_pos, width=20, height=120)
    txt_console.config(yscrollcommand=scr_subsets.set)
    scr_subsets.config(command=txt_console.yview)

    """
    y_pos += 120 + 2
    subject_pb = ttk.Progressbar(win, orient='horizontal', mode='determinate')
    subject_pb.place(x=15, y=y_pos, width=window_width - 30, height=30)
    """

    #
    # apply input/output directories
    #
    if init_input_directory is not None:
        input_directory.set(os.path.abspath(os.path.expanduser(os.path.expandvars(init_input_directory))))

        # provide an initial output directory
        output_dir = output_directory.get()
        if not output_dir:
            output_directory.set(os.path.join(input_directory.get(), 'derivatives', 'erdetect_out'))

        # update the dataset list with the selected folder
        update_datasets(input_directory.get())

    if init_output_directory is not None:
        output_directory.set(os.path.abspath(os.path.expanduser(os.path.expandvars(init_output_directory))))

    #
    # open window
    #
    win.mainloop()
    exit()


if __name__ == "__main__":
    open_gui()
