"""
Definition of useful constants
"""

class DatasetFields:
    CellId = "CellId"
    CellIndex = "CellIndex"
    FOVId = "FOVId"
    SourceReadPath = "SourceReadPath"
    DraftMitoticStateResolved = "Draft mitotic state resolved"
    DraftMitoticStateCoarse = "Draft mitotic state coarse"
    ExpertMitoticStateResolved = "Expert mitotic state resolved"
    ExpertMitoticStateCoarse = "Expert mitotic state coarse"
    DraftM6_7Complete = "Draft M6/M7 complete"
    NucleusSegmentationReadPath = "NucleusSegmentationReadPath"
    MembraneSegmentationReadPath = "MembraneSegmentationReadPath"
    ChannelIndexDNA = "ChannelIndexDNA"
    ChannelIndexMembrane = "ChannelIndexMembrane"
    ChannelIndexStructure = "ChannelIndexStructure"
    ChannelIndexBrightfield = "ChannelIndexBrightfield"
    ChannelIndexNucleusSegmentation = "ChannelIndexNucleusSegmentation"
    ChannelIndexMembraneSegmentation = "ChannelIndexMembraneSegmentation"
    StandardizedFOVPath = "StandardizedFOVPath"
    CellFeaturesPath = "CellFeaturesPath"
    CellImage3DPath = "CellImage3DPath"
    CellImage2DAllProjectionsPath = "CellImage2DAllProjectionsPath"
    CellImage2DYXProjectionPath = "CellImage2DYXProjectionPath"
    Chosen2DProjectionPath = "Chosen2DProjectionPath"
    ChosenMitoticClass = "ChosenMitoticClass"
    DiagnosticSheetPath = "DiagnosticSheetPath"
    ProteinIdName = "ProteinId/Name"
    AllExpectedInputs = [
        CellId,
        CellIndex,
        FOVId,
        CellImage3DPath,
        SourceReadPath,
        NucleusSegmentationReadPath,
        MembraneSegmentationReadPath,
        ChannelIndexDNA,
        ChannelIndexMembrane,
        ChannelIndexStructure,
        ChannelIndexBrightfield,
        ChannelIndexNucleusSegmentation,
        ChannelIndexMembraneSegmentation,
        DraftMitoticStateResolved,
        DraftMitoticStateCoarse,
        ExpertMitoticStateResolved,
        ExpertMitoticStateCoarse,
        DraftM6_7Complete,
    ]
