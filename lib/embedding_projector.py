"""Tensorflow embedding projector to visualize different speaker embeeddings.

----------
Alessandro De Luca - 09/2023
"""
import os
import subprocess
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorboard import program
from tensorboard.plugins import projector


class TensorBoardTool:

    def __init__(self, log_dir: str, embedding_vecs: np.ndarray, metadata: np.ndarray, metadata_var: list):
        """Initialize TensorBoardTool class.

        Args:
            log_dir: Path to log directory for TensorBoard.
            embedding_vecs: Array of embedding vectors.
            metadata: Array of metadata variables. If None, no metadata.
            metadata_vars: Array of metadata variable names. If None, no metadata.
        """
        self.log_dir = log_dir
        self.embedding_vecs = embedding_vecs

        # Make metadata variable name a list if not already
        if metadata is not None and metadata_var is not None:
            if type(metadata_var) is not list:
                self.metadata_var = [metadata_var]
            else:
                self.metadata_var = metadata_var

            # Reshape metadata if 1D
            if len(self.metadata_var) == 1 and len(metadata.shape) == 1:
                self.metadata = metadata.reshape((metadata.shape[0], 1))
            else:
                self.metadata = metadata
        else:
            self.metadata = None
            self.metadata_var = None

    def _data_setup(self) -> None:
        """Prepares data for embedding projector.
        """
        # Make log directory
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Normalize the embeddings
        # self._norm1U() # Did not seem to work all that well; distances between points don't make all that much sense anymore
        # => better to just tick spherize data (this seems to be the case also on the online projector)

        # Create embedding tsv file
        pd.DataFrame(
            data=self.embedding_vecs
        ).astype(np.float64).to_csv(os.path.join(self.log_dir, 'embeddings.tsv'),
                                    sep='\t', header=False, index=False)

        # Metadata file
        if self.metadata is not None:
            if len(self.metadata_var) > 1:
                # Write labels for metadata if there are more than 1 vars
                pd.DataFrame(
                    data=self.metadata, columns=self.metadata_var
                ).to_csv(os.path.join(self.log_dir, 'metadata.tsv'),
                         sep='\t', index=False)
            else:
                # Write only labels
                pd.DataFrame(
                    data=self.metadata
                ).to_csv(os.path.join(self.log_dir, 'metadata.tsv'),
                         sep='\t', index=False, header=False)
        else:
            if os.path.isfile(os.path.join(self.log_dir, 'metadata.tsv')):
                os.remove(os.path.join(self.log_dir, 'metadata.tsv'))

    def _norm1U(self):
        """Normalize the vectors to 1-unit vectors in the Euclidean space.
        This is half of what 'Spherize data' in the embedding projector does
        (it also centers the data), and is done only to fix the bug where UMAP
        remains stuck unless 'Spherize data' is checked.
        The issue arises from kNN needing normalized data as an input.
        """
        self.embedding_vecs = np.array(
            [v / np.linalg.norm(v) for v in self.embedding_vecs])

    def _projector_setup(self) -> None:
        """Sets up embedding projector and projects the embeddings.
        """
        # Prepare data for embedding projector
        self._data_setup()

        # Configure embedding projector
        self.config = projector.ProjectorConfig()
        self.embedding = self.config.embeddings.add()
        self.embedding.tensor_path = 'embeddings.tsv'
        if self.metadata is not None:
            self.embedding.metadata_path = 'metadata.tsv'
        projector.visualize_embeddings(self.log_dir, self.config)

    def run(self):
        """Runs the embedding projector. Returns the TensorBoard object and the URL."""
        self._projector_setup()
        tb = program.TensorBoard()
        tb.configure(argv=[
            None, '--logdir', self.log_dir,
            #'--reuse_port', 'True'
        ])
        url = tb.launch()
        return tb, url+'/#projector'

    def _shutdown(self):
        """Shuts down the TensorBoard server.
        """
        subprocess.Popen(['pkill', '-f', 'tensorboard'])
        logging.info('TensorBoard server shut down.')


def test():
    import pickle
    log_dir = 'for_tensorboard/logs/test/'

    # Load data
    with open('testing/test_mfccs.pkl', 'rb') as f:
        embeddings = pickle.load(f)

    with open('testing/test_sp.pkl', 'rb') as f:
        metadata = pickle.load(f)

    metadata_vars = ['speaker']

    # Run tensorflow embedding projector setup
    tbTool = TensorBoardTool(
        log_dir=log_dir, embedding_vecs=embeddings,
        metadata=None, metadata_var=metadata_vars
    )
    tbTool.run()


if __name__ == '__main__':
    test()
