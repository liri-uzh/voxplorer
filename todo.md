# TODO document for LiRI-DORA.
- [x] pages basic layout
- [x] README
- [!] Check bug issue [issue #2](https://github.com/liri-uzh/voxplorer/issues/2#issue-3681707256) 
- [!] **BUG: plot shape not reset to null after first selection** [issue #3](https://github.com/liri-uzh/voxplorer/issues/3#issue-3920289799) 
- [!] **ERROR: plot before reduction breaks figure** [issue #4](https://github.com/liri-uzh/voxplorer/issues/4#issue-3920728802)   
    - probable fix: don't return a figure when error for no reduced dim table is returned
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
