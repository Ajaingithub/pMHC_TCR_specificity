# pMHC_TCR_specificity
Dataset Description

Test (test.csv) and sample submission (sample_submission.csv) files are provided for competitors to complete with binding probability predictions in a 'Prediction' column indicating a probability between 0 and 1 that the peptide and TCR in a given row will bind. NB: If there are peptides for which your model cannot make predictions, please use a default prediction of zero for these rows. A file of only zero-valued predictions will yield an AUC0.1 of 0.5.

For training data, we provide a sample training dataset from the IEDB1 as well as a special VDJdb release2.
Files

    test.csv - the test set
    sample_submission.csv - a sample submission file in the correct format (test.csv + Prediction column)
    iedb_positives.csv - a sample training dataset of paired TCR-pHLA pairs compiled from the IEDB1.
    vdjdb_positives.csv - a sample training dataset of paired TCR-pHLA pairs compiled from VDJdb2.

Test/submission columns

    ID - an identifier of the TCR-peptide pair used in test and submission files (NB: order matters, do not change the row order or ID values)
    Peptide - the peptide/epitope
    HLA - the HLA/MHC allele name
    Va - the TCR alpha V gene name
    Ja - the TCR alpha J gene name
    TCRa - the TCR alpha amino acid sequence
    CDR1a - the TCR alpha CDR1 amino acid sequence
    CDR2a - the TCR alpha CDR2 amino acid sequence
    CDR3a - the TCR alpha CDR3 amino acid sequence (positions 105-117)
    CDR3a_extended - the TCR alpha CDR3 amino acid sequence (positions 104-118)
    Vb - the TCR beta V gene name
    Jb - the TCR beta J gene name
    TCRb - the TCR beta amino acid sequence
    CDR1b - the TCR beta CDR1 amino acid sequence
    CDR2b - the TCR beta CDR2 amino acid sequence
    CDR3b - the TCR beta CDR3 amino acid sequence (positions 105-117)
    CDR3b_extended - the TCR beta CDR3 amino acid sequence (positions 104-118)
    Prediction - your prediction of the probability of binding [0,1]

Example training data

We include example training data from the IEDB (immrep_IEDB.csv) which includes the above sequence columns with the exception of ID and Prediction, and the additional columns: "receptor_id", "references" and "just_10X" as described.

In addition, VDJdb2 has kindly created a special release with the standard VDJdb format which includes multiple new datasets.
References

1 Vita, Randi, et al. The Immune Epitope Database (IEDB): 2024 update. Nucleic Acids Research 53.D1 (2025): D436-D443. https://doi.org/10.1093/nar/gkae1092
2 Goncharov, M., Bagaev, D., Shcherbinin, D. et al. VDJdb in the pandemic era: a compendium of T cell receptors specific for SARS-CoV-2. Nat Methods 19, 1017–1019 (2022). https://doi.org/10.1038/s41592-022-01578-0

