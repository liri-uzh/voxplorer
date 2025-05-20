# TODO document for LiRI-DORA.
- [x] pages basic layout

## Visualiser
    - [-] table view page functions and callbacks
    - [ ] plot view page functions and callbacks
    - [-] interconnectivity across pages (persistance of data + filtering on same data)
    - [ ] feature extraction and data stream
### Layouts
    - [ ] Options card:
        - if audio upload:
            -> feature extraction options
            -> select meta-information
        x if table upload:
            x -> select meta-information
    - [x] Table preview + filtering and selection/deselection
    - [ ] Plot options card:
        - dim-reduction selection
        - dim-reduction specific options
    - [ ] Plot view + col / shape selection

## Recogniser
    - [ ] model selection
    - [ ] implementation of reference population for LR with ECAPA-TDNN
    - [ ] implementation of UBM training
    - [ ] results reporting and download
### Layouts
    - [ ] model selection options cards:
        - if ECAPA: upload reference pop.
        - if GMM-UBM: upload training pop.
    - [ ] comparison list upload (list of filenames in pairs with separator indication)
    - [ ] Run button (appears only if everything has been uploaded)
    - [ ] Progress info boxes (training and inference)
    - [ ] Plot view
    - [ ] download results button (available after finished running)
