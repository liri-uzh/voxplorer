"""
Graphical User Interface for voxplorer built using tkinter.
"""

from collections import defaultdict
import os
import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog, ttk
import threading
from typing import Callable
import webbrowser
import json
import numpy as np
import pandas as pd
from lib import data_manager, embedding_projector
from lib.feature_extraction import FeatureExtractor
from lib.speaker_embeddings import SpeakerEmbedder


# TODO: test with excel files


class VisualizerWindow(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title('Visualizer mode')
        self.create_widgets()

    def create_widgets(self):
        file_frame = tk.Frame(self)
        file_frame.pack()

        self.file_label = tk.Label(file_frame, text='Table path:')
        self.file_label.grid(row=0, column=0, columnspan=2)

        self.file_entry = tk.Entry(file_frame)
        self.file_entry.grid(row=1, column=0, sticky='w')

        self.file_button = tk.Button(
            file_frame, text='Browse', command=self.select_file)
        self.file_button.grid(row=1, column=1, sticky='e')

        outer_frame = tk.Frame(self)
        outer_frame.pack()

        self.use_json_dtypes_var = tk.IntVar()
        self.use_json_dtypes_check = tk.Checkbutton(
            outer_frame, text="Use JSON dtypes", variable=self.use_json_dtypes_var)
        self.use_json_dtypes_check.pack()

        self.columns_label = tk.Label(
            outer_frame, text='Select columns for metadata features:')
        self.columns_label.pack()

        self.columns_listbox = tk.Listbox(outer_frame, selectmode='multiple')
        self.columns_listbox.pack()

        self.selection_var = tk.IntVar()
        self.selection_check = tk.Checkbutton(
            outer_frame, text="Add 'selection' column", variable=self.selection_var)
        self.selection_check.pack()

        self.continue_button = tk.Button(
            outer_frame, text='Continue', command=self.open_projector)
        self.continue_button.pack()

    def select_file(self):
        file_path = filedialog.askopenfilename(
            initialdir='.', title='Select a table')
        if file_path:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, file_path)
            self.load_columns(file_path)

    def load_columns(self, file_path):
        try:
            table = data_manager.load_data(file_path)
        except UnicodeDecodeError:
            messagebox.showerror(
                "Error", "Cannot read the file. Please check the encoding.")
            return

        self.columns_listbox.delete(0, tk.END)  # clear the listbox
        for col in table.columns:
            self.columns_listbox.insert(tk.END, col)

    def open_projector(self):
        if not self.file_entry.get():
            messagebox.showerror(
                "Error", "Please select a data table.")
            return

        # Get metavars
        selected_indices = self.columns_listbox.curselection()
        selected_columns = [self.columns_listbox.get(
            i) for i in selected_indices]
        add_selection_column = bool(self.selection_var.get())
        use_json_dtypes = bool(self.use_json_dtypes_var.get())

        # Load data
        df = data_manager.load_data(
            path_to_data=self.file_entry.get(),
            metavars=selected_columns,
            use_json_dtypes=use_json_dtypes)
        X, Y, metavars = data_manager.split_data(
            df=df, features=None, metavars=selected_columns,
            add_selection_column=add_selection_column
        )

        # Start projector in window
        ProjectorWindow(self, X, Y, metavars)


class FeatureExtractorWindow(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title('Feature extraction mode')
        self.create_widgets()

    def create_widgets(self):
        file_frame = tk.Frame(self)
        file_frame.pack()

        self.file_label = tk.Label(
            file_frame, text='Directory or WAV file path:')
        self.file_label.grid(row=0, column=0, columnspan=3)

        self.file_entry = tk.Entry(file_frame)
        self.file_entry.grid(row=1, column=0, sticky='w')

        self.file_button = tk.Button(
            file_frame, text='Browse file', command=self.select_file)
        self.file_button.grid(row=1, column=1, sticky='e')

        self.dir_button = tk.Button(
            file_frame, text='Browse directory', command=self.select_dir)
        self.dir_button.grid(row=1, column=2, sticky='e')

        self.out_dir_label = tk.Label(
            file_frame, text='Output directory & Filename:')
        self.out_dir_label.grid(row=2, column=0, columnspan=3)

        self.out_dir_entry = tk.Entry(file_frame)
        self.out_dir_entry.grid(row=3, column=0, sticky='w')

        self.out_file_entry = tk.Entry(file_frame)
        self.out_file_entry.grid(row=3, column=1, sticky='e')

        self.out_dir_button = tk.Button(
            file_frame, text='Browse', command=self.select_out_dir)
        self.out_dir_button.grid(row=3, column=2, sticky='e')

        outer_frame = tk.Frame(self)
        outer_frame.pack()

        self.mode_var = tk.StringVar(value="speaker embeddings")
        self.speaker_embeddings_radio = tk.Radiobutton(
            outer_frame, text="Speaker embeddings", variable=self.mode_var,
            value="speaker embeddings", command=self.update_methods_listbox)
        self.speaker_embeddings_radio.pack()

        self.feature_extraction_radio = tk.Radiobutton(
            outer_frame, text="Feature extraction", variable=self.mode_var,
            value="feature extraction", command=self.update_methods_listbox
        )
        self.feature_extraction_radio.pack()

        self.methods_label = tk.Label(
            outer_frame, text='Select feature extraction methods:')
        self.methods_listbox = tk.Listbox(self, selectmode='multiple')
        for method in [
            "Mel Features",
            "Acoustic Features (Pitch desc., Formants, VT estimates, HNR, jitter, shimmer, RMS energy)",
            "Low Level Features (spectral centroid, bandwidth, contrasts, flatness, rolloff, zero-crossing rate)",
            "Liner Predictive Cepstral Coefficients"
        ]:
            self.methods_listbox.insert(tk.END, method)

        self.continue_button = tk.Button(outer_frame, text='Continue',
                                         command=self.open_methods_window)
        self.continue_button.pack()

    def update_methods_listbox(self):
        if self.mode_var.get() == "feature extraction":
            self.methods_label.pack(padx=10, pady=10)
            self.methods_listbox.pack(
                fill=tk.BOTH, expand=True, padx=10, pady=10)
        else:
            self.methods_label.pack_forget()
            self.methods_listbox.pack_forget()

    def select_file(self):
        file_path = filedialog.askopenfilename(
            initialdir='.', title='Select a directory of WAV files or a WAV file')
        if file_path:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, file_path)

    def select_dir(self):
        dir_path = filedialog.askdirectory(
            initialdir='.', title='Select a directory of WAV files or a WAV file')
        if dir_path:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, dir_path)

    def select_out_dir(self):
        dir_path = filedialog.askdirectory(
            initialdir='.', title='Select a directory')
        if dir_path:
            self.out_dir_entry.delete(0, tk.END)
            self.out_dir_entry.insert(0, dir_path)

    def open_methods_window(self):
        if not self.file_entry.get():
            messagebox.showerror(
                "Error", "Please select a directory containing WAV files or a WAV file.")
            return

        if self.mode_var.get() == "feature extraction":
            self.feature_methods = {}
            self.method_translator = {
                "Mel Features": "mel_features",
                "Acoustic Features (Pitch desc., Formants, VT estimates, HNR, jitter, shimmer, RMS energy)": "acoustic_features",
                "Low Level Features (spectral centroid, bandwidth, contrasts, flatness, rolloff, zero-crossing rate)": "low_lvl_features",
                "Liner Predictive Cepstral Coefficients": "lpc_features"
            }
            selected_indices = self.methods_listbox.curselection()
            selected_methods = [self.methods_listbox.get(
                i) for i in selected_indices]
            MethodsWindow(self, selected_methods, self.store_parameter_values)

        else:
            self.filename = self.file_entry.get()
            if os.path.isdir(self.filename):
                for f in os.listdir(self.filename):
                    if f.endswith('.wav'):
                        filename = os.path.splitext(os.path.basename(f))[0]
                        break
            else:
                filename = os.path.splitext(os.path.basename(self.filename))[0]
            self.metadata_vars = {'metavars': None, 'separator': None}
            MetadataWindow(self, filename, self.store_metadata_specifications)

    def store_parameter_values(self, method, parameter_values):
        self.feature_methods[self.method_translator[method]] = parameter_values

        # Check if all methods have been processed
        if len(self.feature_methods) == len(self.methods_listbox.curselection()):
            self.filename = self.file_entry.get()
            if os.path.isdir(self.filename):
                for f in os.listdir(self.filename):
                    if f.endswith('.wav'):
                        filename = os.path.splitext(os.path.basename(f))[0]
                        break
            else:
                filename = os.path.splitext(os.path.basename(self.filename))[0]
            self.metadata_vars = {'metavars': None, 'separator': None}
            MetadataWindow(self, filename, self.store_metadata_specifications)

    def store_metadata_specifications(self, metavars: list, separator: str, add_selection_column: bool):
        self.metadata_vars['metavars'] = metavars
        self.metadata_vars['separator'] = separator
        self.metadata_vars['add_selection_column'] = add_selection_column

        # Check if the start analysis button already exists
        if hasattr(self, 'start_analysis_button'):
            # Update command of existing button
            self.start_analysis_button['command'] = self.feature_extraction_pipeline
        else:
            # Add a start analysis button
            self.start_analysis_button = tk.Button(
                self, text='Start analysis', command=self.feature_extraction_pipeline)
            self.start_analysis_button.pack()

        self.deiconify()

    def feature_extraction_pipeline(self):
        if self.mode_var.get() == "feature extraction":
            featureExtractor = FeatureExtractor(
                audio_dir=self.filename, feature_methods=self.feature_methods,
                metadata_vars=self.metadata_vars
            )
        else:
            featureExtractor = SpeakerEmbedder(
                audio_dir=self.filename, metadata_vars=self.metadata_vars
            )

        X, Y, metavars, feature_labels = featureExtractor.process_files()

        if self.out_dir_entry.get():
            out_dir = self.out_dir_entry.get()
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            if self.out_file_entry.get():
                out_file = os.path.splitext(self.out_file_entry.get())[0]
            else:
                out_file = 'features'

            # Output dataframe
            df_Y = pd.DataFrame(data=Y, columns=metavars,
                                dtype=pd.api.types.CategoricalDtype())

            df_X = pd.DataFrame(data=X, columns=feature_labels)

            df = pd.concat([df_Y, df_X], axis=1)

            df.to_csv(os.path.join(out_dir, out_file + '.csv'), index=False)
            df.dtypes.apply(lambda x: x.name).to_json(
                os.path.join(out_dir, out_file + '_dtypes.json'))

            messagebox.showinfo(
                "Success", "Features saved to " +
                os.path.join(out_dir, out_file + '.csv')
            )

        else:
            messagebox.showinfo(
                "Success", "Feature extraction finished."
            )

        # Start projector in window
        ProjectorWindow(self, X, Y, metavars)


class MethodsWindow(tk.Toplevel):
    def __init__(self, master=None, methods: list = None, callback: Callable = None, index: int = 0):
        super().__init__(master)
        self.title('Feature extraction')
        self.methods = methods
        self.callback = callback
        self.index = index
        self.method = methods[index]
        self.parameters = {}
        self.create_widgets()

    def create_widgets(self):
        self.method_label = tk.Label(
            self, text="Set parameters for " + self.method)
        self.method_label.grid(row=0, column=0, columnspan=2)

        # Define parameters for each method
        parameters = {
            "Mel Features": {
                "n_mfcc": 13,
                "n_mels": 40,
                "win_length": 25.0,
                "overlap": 10.0,
                "fmin": 150.0,
                "fmax": 4000.0,
                "premphasis": 0.95,
                "lifter": 22.0,
                "deltas": True,
                "summarise": True
            },
            "Acoustic Features (Pitch desc., Formants, VT estimates, HNR, jitter, shimmer, RMS energy)": {
                "f0min": 75.0, "f0max": 600.0
            },
            "Low Level Features (spectral centroid, bandwidth, contrasts, flatness, rolloff, zero-crossing rate)": {
                "win_length": 25.0,
                "overlap": 10.0,
                "premphasis": 0.95,
                "n_bands_contrasts": 6,
                "use_mean_contrasts": True,
                "summarise": True
            },
            "Liner Predictive Cepstral Coefficients": {
                "n_lpcc": 13,
                "win_length": 25.0,
                "overlap": 10.0,
                "premphasis": 0.95,
                "order": 16,
                "summarise": True
            }
        }

        for i, (parameter, default_value) in enumerate(parameters[self.method].items()):
            label = tk.Label(self, text=parameter, width=20, anchor='w')
            label.grid(row=i+1, column=0, sticky='w')

            # Checkbutton for boolean parameters
            if isinstance(default_value, bool):
                var = tk.BooleanVar(value=default_value)
                widget = tk.Checkbutton(self, variable=var)
            elif isinstance(default_value, float):
                var = tk.DoubleVar(value=default_value)
                widget = tk.Entry(self, textvariable=var)
            else:
                var = tk.IntVar(value=default_value)
                widget = tk.Entry(self, textvariable=var)
            widget.grid(row=i+1, column=1, sticky='e')

            # Store variable in the parameter dictionary and default value for type checking
            self.parameters[parameter] = (var, default_value)

        self.continue_button = tk.Button(
            self, text='Continue', command=self.save_parameters)
        self.continue_button.grid(
            row=len(parameters[self.method])+1, column=0, columnspan=2)

    def save_parameters(self):
        # Save parameters to a dictionary
        parameters_dict = {}
        for parameter, (var, default_value) in self.parameters.items():
            try:
                parameters_dict[parameter] = var.get()
            except tk.TclError:
                messagebox.showerror(
                    "Error", "Invalid value for " + parameter + ".")
                var.set(default_value)
                return
        if self.callback is not None:
            self.callback(self.method, parameters_dict)
        self.index += 1
        if self.index < len(self.methods):
            # Open the next MethodsWindow
            MethodsWindow(self.master, self.methods,
                          self.callback, self.index)
        self.destroy()  # close the current window

    def destroy(self) -> None:
        self.withdraw()


class MetadataWindow(tk.Toplevel):
    def __init__(self, master=None, filename: str = None, callback: Callable = None):
        super().__init__(master)
        self.title('Metadata specification')
        self.filename = filename
        self.separator = tk.StringVar(value='_')
        self.callback = callback
        self.separator.trace('w', self.update_filename_parts)
        self.create_widgets()

    def create_widgets(self):
        separator_frame = tk.Frame(self)
        separator_frame.pack()

        self.separator_label = tk.Label(separator_frame, text='Separator:')
        self.separator_label.grid(row=0, column=0, sticky='w')

        self.separator_value = tk.Entry(
            separator_frame, textvariable=self.separator)
        self.separator_value.grid(row=0, column=1, sticky='e')

        self.filename_parts_frame = tk.Frame(self)
        self.filename_parts_frame.pack()

        outer_frame = tk.Frame(self)
        outer_frame.pack()

        self.selection_var = tk.IntVar()
        self.selection_check = tk.Checkbutton(
            outer_frame, text="Add 'selection' column", variable=self.selection_var)
        self.selection_check.pack()

        self.continue_button = tk.Button(
            outer_frame, text='Continue', command=self.save_metadata)
        self.continue_button.pack()

        # Call update_filename_parts to display the filename parts with default separator
        self.update_filename_parts()

    def update_filename_parts(self, *args):
        for widget in self.filename_parts_frame.winfo_children():
            widget.destroy()

        if self.separator.get() != '':
            self.current_separator = self.separator.get()
        filename_parts = self.filename.split(self.current_separator)
        metadata_spec_label = tk.Label(
            self.filename_parts_frame, text='Metadata specification:')
        metadata_spec_label.grid(row=0, column=0, columnspan=2)
        for i, part in enumerate(filename_parts):
            label = tk.Label(self.filename_parts_frame, text=part)
            label.grid(row=i + 1, column=0, sticky='w')

            entry = tk.Entry(self.filename_parts_frame)
            entry.grid(row=i + 1, column=1, sticky='e')

    def save_metadata(self):
        metadata_spec = []
        for widget in self.filename_parts_frame.winfo_children():
            if isinstance(widget, tk.Entry):
                metadata_spec.append(widget.get())
        add_selection_column = bool(self.selection_var.get())
        if self.callback is not None:
            self.callback(metadata_spec, self.current_separator,
                          add_selection_column)
        self.destroy()

    def destroy(self) -> None:
        self.withdraw()


class FilterWindow(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title('Filtering mode')
        self.create_widgets()

    def create_widgets(self):
        file_frame = tk.Frame(self)
        file_frame.pack()

        self.file_label = tk.Label(file_frame, text='Table path:')
        self.file_label.grid(row=0, column=0, columnspan=2)

        self.file_entry = tk.Entry(file_frame)
        self.file_entry.grid(row=1, column=0, sticky='w')

        self.file_button = tk.Button(
            file_frame, text='Browse', command=self.select_file)
        self.file_button.grid(row=1, column=1, sticky='e')

        self.mod_metadata_label = tk.Label(
            file_frame, text='Modified metadata table path:')
        self.mod_metadata_label.grid(row=2, column=0, columnspan=2)

        self.mod_metadata_entry = tk.Entry(file_frame)
        self.mod_metadata_entry.grid(row=3, column=0, sticky='w')

        self.mod_metadata_button = tk.Button(
            file_frame, text='Browse', command=self.select_mod_metadata)
        self.mod_metadata_button.grid(row=3, column=1, sticky='e')

        self.mod_metadata_joinkey_label = tk.Label(
            file_frame, text='Join key if using modified metadata table:')
        self.mod_metadata_joinkey_label.grid(row=4, column=0, columnspan=2)

        self.mod_metadata_joinkey_entry = tk.Entry(file_frame)
        self.mod_metadata_joinkey_entry.grid(row=5, column=0, columnspan=2)

        self.filter_var_label = tk.Label(
            file_frame, text='Filter variable:')
        self.filter_var_label.grid(row=6, column=0, sticky='w')

        self.filter_entry = tk.Entry(file_frame)
        self.filter_entry.grid(row=6, column=1, sticky='e')

        outer_frame = tk.Frame(self)
        outer_frame.pack()

        self.use_json_dtypes_var = tk.IntVar()
        self.use_json_dtypes_check = tk.Checkbutton(
            outer_frame, text="Use JSON dtypes", variable=self.use_json_dtypes_var)
        self.use_json_dtypes_check.pack()

        self.columns_label = tk.Label(
            outer_frame, text='Select columns for metadata features:')
        self.columns_label.pack()

        self.columns_listbox = tk.Listbox(outer_frame, selectmode='multiple')
        self.columns_listbox.pack()

        self.continue_button = tk.Button(
            outer_frame, text='Continue', command=self.open_filter_window)
        self.continue_button.pack()

    def select_file(self):
        file_path = filedialog.askopenfilename(
            initialdir='.', title='Select a table')
        if file_path:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, file_path)
            self.load_columns(file_path)

    def select_mod_metadata(self):
        file_path = filedialog.askopenfilename(
            initialdir='.', title='Select a table')
        if file_path:
            self.mod_metadata_entry.delete(0, tk.END)
            self.mod_metadata_entry.insert(0, file_path)

    def load_columns(self, file_path):
        try:
            table = data_manager.load_data(file_path)
        except UnicodeDecodeError:
            messagebox.showerror(
                "Error", "Cannot read the file. Please check the encoding.")
            return

        self.columns_listbox.delete(0, tk.END)  # clear the listbox
        for col in table.columns:
            self.columns_listbox.insert(tk.END, col)

    def open_filter_window(self):
        if not self.file_entry.get():
            messagebox.showerror(
                "Error", "Please select a data table.")
            return

        # Get metavars
        selected_indices = self.columns_listbox.curselection()
        selected_columns = [self.columns_listbox.get(
            i) for i in selected_indices]
        use_json_dtypes = bool(self.use_json_dtypes_var.get())

        # Load table
        df = data_manager.load_data(
            path_to_data=self.file_entry.get(),
            metavars=selected_columns,
            use_json_dtypes=use_json_dtypes
        )

        if not self.filter_entry.get():
            messagebox.showerror(
                "Error", "Please provide a filter variable.")
            return

        # Load modified metadata
        if self.mod_metadata_entry.get():
            if self.use_json_dtypes_var.get():
                filename = os.path.splitext(self.file_entry.get())[0]
                try:
                    with open(os.path.join(filename + '_dtypes.json'), 'r') as f:
                        dtypes = json.load(f)
                    dtypes = {k: pd.api.types.pandas_dtype(
                        v) for k, v in dtypes.items()}
                except FileNotFoundError:
                    messagebox.showerror(
                        "Error", "Cannot find dtypes file. This should be saved in the same folder as the data file with the extension _dtypes.json.")
                    return
                df_mod = pd.read_csv(
                    self.mod_metadata_entry.get(), dtype=dtypes, sep='\t')
            else:
                df_mod = pd.read_csv(self.mod_metadata_entry.get(), sep='\t')

            if self.mod_metadata_joinkey_entry.get():
                join_key = self.mod_metadata_joinkey_entry.get()
                df_mod = df_mod.drop(columns=df_mod.columns.drop(
                    [join_key, self.filter_entry.get()]))
                if self.filter_entry.get() in df.columns:
                    df = df.drop(columns=self.filter_entry.get())
                print(df.columns)
                print(df_mod.columns)
            else:
                messagebox.showwarning(
                    "Warning", "No join key provided. Joining using intersection of columns."
                )
                join_key = None
                df_mod = df_mod.drop(
                    columns=df_mod.columns.drop(self.filter_entry.get()))
                if self.filter_entry.get() in df.columns:
                    df = df.drop(columns=self.filter_entry.get())

            df = pd.merge(df_mod, df, on=join_key)

        # Open filter selection window
        FilterSelectionWindow(self, df=df, filter_var=self.filter_entry.get())


class FilterSelectionWindow(tk.Toplevel):
    def __init__(self, master=None, df: pd.DataFrame = None, filter_var: str = None):
        super().__init__(master)
        self.title('Filter selection')
        self.filter_var = filter_var
        self.df = df
        self.create_widgets()

    def create_widgets(self):
        file_frame = tk.Frame(self)
        file_frame.pack()

        self.out_dir_label = tk.Label(
            file_frame, text='Output directory & Filename:'
        )
        self.out_dir_label.grid(row=0, column=0, columnspan=3)

        self.out_dir_entry = tk.Entry(file_frame)
        self.out_dir_entry.grid(row=1, column=0, sticky='w')

        self.out_file_entry = tk.Entry(file_frame)
        self.out_file_entry.grid(row=1, column=1, sticky='e')

        self.out_dir_button = tk.Button(
            file_frame, text='Browse', command=self.select_out_dir)
        self.out_dir_button.grid(row=1, column=2, sticky='e')

        outer_frame = tk.Frame(self)
        outer_frame.pack()

        self.filter_label = tk.Label(
            outer_frame, text=f'Filter value for {self.filter_var}:'
        )
        self.filter_label.pack()

        self.filter_entry = tk.Entry(outer_frame)
        self.filter_entry.pack()

        self.continue_button = tk.Button(
            outer_frame, text='Filter', command=self.filter_selection)
        self.continue_button.pack()

    def select_out_dir(self):
        dir_path = filedialog.askdirectory(
            initialdir='.', title='Select a directory')
        if dir_path:
            self.out_dir_entry.delete(0, tk.END)
            self.out_dir_entry.insert(0, dir_path)

    def filter_selection(self):
        if not self.out_dir_entry.get():
            messagebox.showerror(
                "Error", "Please select an output directory.")
            return
        else:
            if not os.path.exists(self.out_dir_entry.get()):
                os.mkdir(self.out_dir_entry.get())
        if self.out_file_entry.get():
            out_file = os.path.splitext(self.out_file_entry.get())[0]
        else:
            out_file = 'filtered_selection'

        # Filter to correct format
        filter_value = self.filter_entry.get()

        if not type(filter_value) is str:
            filter_value = str(filter_value)

        # Filter the selection partition
        data_manager.filter_selection(
            df=self.df,
            output_file=os.path.join(
                self.out_dir_entry.get(), out_file + '.csv'),
            metavar=self.filter_var,
            metavar_filter=filter_value
        )
        messagebox.showinfo(
            "Success", "Filtered selection saved to " +
            os.path.join(self.out_dir_entry.get(), out_file + '.csv')
        )
        self.destroy()


class ProjectorWindow(tk.Toplevel):
    def __init__(self, master=None, X: np.ndarray = None, Y: np.ndarray = None, metavars: list = None):
        super().__init__(master)
        self.title('Projector')
        self.X = X
        self.Y = Y
        self.metavars = metavars
        self.create_widgets()

    def create_widgets(self):
        self.projector_label = tk.Label(self, text='Projector')
        self.projector_label.pack()

        self.project_button = tk.Button(
            self, text='Project!', command=self.start_projector)
        self.project_button.pack()

    def start_projector(self):
        # Start projector in separate thread
        self.projector_thread = threading.Thread(
            target=self.embedding_projector, args=())
        self.projector_thread.start()

        # Add button to stop analysis
        self.stop_button = tk.Button(self, text='Stop projector',
                                     command=self.stop_projector)
        self.stop_button.pack()

    def embedding_projector(self):
        log_dir = os.path.join(os.getcwd(), 'logs')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        # Run embedding projector
        self.tbTool = embedding_projector.TensorBoardTool(
            log_dir=log_dir, embedding_vecs=self.X, metadata=self.Y,
            metadata_var=self.metavars
        )

        self.board, url = self.tbTool.run()
        webbrowser.open(url)

    def stop_projector(self):
        # if self.projector_thread:
        #     self.projector_thread.terminate()
        #     self.board = None
        #     self.stop_button.destroy()
        if self.board:
            self.tbTool._shutdown()
            self.board = None
            self.stop_button.destroy()


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title('voxplorer')
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.welcome = tk.Label(self, text='Welcome to voxplorer!')
        self.welcome.pack(side='top')

        self.visualizer_button = tk.Button(self)
        self.visualizer_button['text'] = 'Visualizer mode'
        self.visualizer_button['command'] = self.visualizer_mode

        self.visualizer_button.pack()

        self.feature_extractor_button = tk.Button(self)
        self.feature_extractor_button['text'] = 'Feature extraction and visualization mode'
        self.feature_extractor_button['command'] = self.feature_extractor_mode
        self.feature_extractor_button.pack()

        self.filter_mode_button = tk.Button(self)
        self.filter_mode_button['text'] = 'Filtering mode'
        self.filter_mode_button['command'] = self.filter_mode
        self.filter_mode_button.pack()

    def visualizer_mode(self):
        self.visualizer_window = VisualizerWindow(self)

    def feature_extractor_mode(self):
        self.feature_extractor_window = FeatureExtractorWindow(self)

    def filter_mode(self):
        self.filter_window = FilterWindow(self)


def main():
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()
