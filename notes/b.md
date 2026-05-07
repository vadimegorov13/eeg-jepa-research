WHat I've done so far:
  Trained multiple datasets on the pretrained and random weights.
  They all have different monatge, number of electrodes, frequency, and number of trials

  Looked into the spatial layer

What can be done next:
  Spatial adaptation bottleneck in cross-dataset S-JEPA transfer for motor imagery EEG

Released S-JEPA weights provide useful MI representations for some datasets, but naive transfer is not robust. Successful transfer is associated with stable learning in the PreLocal spatial convolution layer, while failure cases show collapse or weak spatial adaptation.

Need to finish the diagnostic story:
  pretrained vs random
  CSP/FBCSP baseline
  spatial_conv diagnostics
  window sensitivity

Future Work:
  montage-aware adapter
  target-domain S-JEPA adaptation
  multi-dataset pretraining