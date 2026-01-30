# TODO document for LiRI-DORA.
- [x] pages basic layout
- [x] README
- [ ] clean tests and dirs from deprecated modules
- [ ] Test windows installation
- [ ] Make sure upload first as str and then modify metavars to str and others to floating point


## Visualiser
    - [x] table view page functions and callbacks
    - [x] plot view page functions and callbacks
    - [x] interconnectivity across pages (persistance of data + filtering on same data)
    - [x] **feature extraction and data stream**
### Layouts
    - [x] Options card:
        - if audio upload:
            -> ~feature extraction options~
            -> ~select meta-information~
        - if table upload:
            -> ~select meta-information~
    - [x] Table preview + filtering and selection/deselection
    - [x] Plot options card:
        - ~dim-reduction selection~
        - ~dim-reduction specific options~
    - [x] Plot view + col / shape selection

## ~Recogniser~
    - [ ] model selection
    - [ ] implementation of reference population for LR with ECAPA-TDNN?
    - [ ] implementation of UBM training?
    - [ ] results reporting and download
### Layouts
    - [ ] model selection options cards:
        - if ECAPA: upload reference pop.?
        - if GMM-UBM: upload training pop.?
    - [ ] comparison list upload (list of filenames in pairs with separator indication)
    - [ ] Run button (appears only if everything has been uploaded)
    - [ ] Progress info boxes (training and inference)
    - [ ] Plot ?view
    - [ ] download results button (available after finished running)
