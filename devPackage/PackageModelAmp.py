import pandas as pd
from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor
import re


class EncodeModelAmp:
    def __init__(self, dictionary, featureDict):
        self.featureDf = None
        self.sequencelist = list(dictionary.values())
        self.seqsName = list(dictionary.keys())
        self.featureDict = featureDict
        self.encode()

    def encode(self):
        """
        If default = True which will output the # of mark:
            #Length : Method to calculate the length (total AA count) of every sequence in the attribute sequences.
            #Calculate_mw : Method to calculate the molecular weight [g/mol] of every sequence in the attribute sequences.
            #Calculate_charge : Method to overall charge of every sequence in the attribute sequences.
            #Chargedensity : Method to calculate the charge density (charge / MW) of every sequences in the attributes sequences.
            #Isoelectric_Point : Method to calculate the isoelectric point of every sequence in the attribute sequences.
            #Instability_Index : Method to calculate the instability of every sequence in the attribute sequences.
            #Aromaticity : Method to calculate the aromaticity of every sequence in the attribute sequences.
            #Aliphatic_Index : Method to calculate the aliphatic index of every sequence in the attribute sequences.
            #Hydrophobic : Method to calculate the hydrophobic ratio of every sequence in the attribute sequences, which is the relative frequency of the amino acids A,C,F,I,L,M & V.
            #AASI : An amino acid selectivity index scale for helical antimicrobial peptides.
            #Argos : Argos hydrophobicity amino acid scale.
            #Bulkiness : Amino acid side chain bulkiness scale.
            #Charge_phys : Amino acid charge at pH 7.0 - Hystidine charge +0.1.
            #Charge_acid : Amino acid charge at acidic pH - Hystidine charge +1.0.
            #Flexibility : Amino acid side chain flexibilitiy scale.
            #Gravy : GRAVY hydrophobicity amino acid scale.
            #Levitt_alpha : Levitt amino acid alpha-helix propensity scale.
            #MSS : A graph-theoretical index that reflects topological shape and size of amino acid side chains.
            #Polarity : Amino acid polarity scale.
            #Refractivity : Relative amino acid refractivity values.
            #TM_tend : Amino acid transmembrane propensity scale.
            Formula : Method to calculate the molecular formula of every sequence in the attribute sequences.
            Boman_Index : Method to calculate the boman index of every sequence in the attribute sequences.
            Eisenberg : The Eisenberg hydrophobicity consensus amino acid scale.
            Hopp_woods : Hopp-Woods amino acid hydrophobicity scale.
            Janin : Janin hydrophobicity amino acid scale.
            Kytedoolittle : Kyte & Doolittle hydrophobicity amino acid scale.
            ABHPRK : modlabs inhouse physicochemical feature scale (Acidic, Basic, Hydrophobic, Polar, aRomatic, Kink-inducer)
            cougar : modlabs inhouse selection of global peptide descriptors
            Ez : potential that assesses energies of insertion of amino acid side chains into lipid bilayers
            MSW : Amino acid scale based on a PCA of the molecular surface based WHIM descriptor (MS-WHIM)
            pepArc : modlabs pharmacophoric feature scale, dimensions are: hydrophobicity, polarity, positive charge, negative charge, proline.
            z3 : The original three dimensional Z-scale
            z5 : The extended five dimensional Z-scale
        :return: DataFrame of Feature
        """
        featureDf = pd.DataFrame(self.seqsName)
        if self.featureDict.get("length") is True:
            lengthSequence = GlobalDescriptor(self.sequencelist)
            lengthSequence.length()
            peptideLength = lengthSequence.descriptor
            peptideLengthDf = pd.DataFrame(peptideLength)
            featureDf.insert(len(featureDf.columns.tolist()), 'Length', peptideLengthDf)

        if self.featureDict["calculate_mw"][0] is True:
            mwSequence = GlobalDescriptor(self.sequencelist)
            mwSequence.calculate_MW(amide=self.featureDict["calculate_mw"][1])
            peptideCalculate_MW = mwSequence.descriptor
            peptideCalculate_MWDf = pd.DataFrame(peptideCalculate_MW)
            featureDf.insert(len(featureDf.columns.tolist()), 'Calculate_mw', peptideCalculate_MWDf)

        if self.featureDict["calculate_charge"][0] is True:
            chargeSequence = GlobalDescriptor(self.sequencelist)
            chargeSequence.calculate_charge(ph=self.featureDict["calculate_charge"][1], amide=self.featureDict["calculate_charge"][2])
            peptideCalculate_Charge = chargeSequence.descriptor
            peptideCalculate_ChargeDf = pd.DataFrame(peptideCalculate_Charge)
            featureDf.insert(len(featureDf.columns.tolist()), 'Calculate_charge', peptideCalculate_ChargeDf)

        if self.featureDict["charge_density"][0] is True:
            densitySequence = GlobalDescriptor(self.sequencelist)
            densitySequence.charge_density(ph=self.featureDict["charge_density"][1], amide=self.featureDict["charge_density"][2])
            peptideDensity = densitySequence.descriptor
            peptideDensityDf = pd.DataFrame(peptideDensity)
            featureDf.insert(len(featureDf.columns.tolist()), 'Charge_density', peptideDensityDf)

        if self.featureDict["isoelectric_point"][0] is True:
            isolectricSequence = GlobalDescriptor(self.sequencelist)
            isolectricSequence.isoelectric_point(amide=self.featureDict["isoelectric_point"][1])
            peptideIsolectric_Point = isolectricSequence.descriptor
            peptideIsolectric_PointDf = pd.DataFrame(peptideIsolectric_Point)
            featureDf.insert(len(featureDf.columns.tolist()), 'Isoelectric_point', peptideIsolectric_PointDf)

        if self.featureDict.get("instability_index") is True:
            instabilitySequence = GlobalDescriptor(self.sequencelist)
            instabilitySequence.instability_index()
            peptideInstability_Index = instabilitySequence.descriptor
            peptideInstability_IndexDf = pd.DataFrame(peptideInstability_Index)
            featureDf.insert(len(featureDf.columns.tolist()), 'Instability_index', peptideInstability_IndexDf)

        if self.featureDict.get("aromaticity") is True:
            aromaticitySequence = GlobalDescriptor(self.sequencelist)
            aromaticitySequence.aromaticity()
            peptideAromaticity = aromaticitySequence.descriptor
            peptideAromaticityDf = pd.DataFrame(peptideAromaticity)
            featureDf.insert(len(featureDf.columns.tolist()), 'Aromaticity', peptideAromaticityDf)

        if self.featureDict.get("aliphatic_index") is True:
            alicphaticSequence = GlobalDescriptor(self.sequencelist)
            alicphaticSequence.aliphatic_index()
            peptideAliphatic_Index = alicphaticSequence.descriptor
            peptideAliphatic_IndexDf = pd.DataFrame(peptideAliphatic_Index)
            featureDf.insert(len(featureDf.columns.tolist()), 'Aliphatic_Index', peptideAliphatic_IndexDf)

        if self.featureDict.get("hydrophobic") is True:
            hydrophobicSequence = GlobalDescriptor(self.sequencelist)
            hydrophobicSequence.hydrophobic_ratio()
            peptideHydrophobic = hydrophobicSequence.descriptor
            peptideHydrophobicDf = pd.DataFrame(peptideHydrophobic)
            featureDf.insert(len(featureDf.columns.tolist()), 'Hydrophobic', peptideHydrophobicDf)

        if self.featureDict.get("aasi") is True:
            AASISequence = PeptideDescriptor(self.sequencelist, "AASI")
            AASISequence.calculate_moment()
            peptideAASI = AASISequence.descriptor
            peptideAASIDf = pd.DataFrame(peptideAASI)
            featureDf.insert(len(featureDf.columns.tolist()), 'AASI', peptideAASIDf)

        if self.featureDict["abhprk"][0] is True:
            abhprkSequence = PeptideDescriptor(self.sequencelist, "ABHPRK")
            abhprkSequence.calculate_crosscorr(window=self.featureDict["abhprk"][1])
            peptideAbhprk = abhprkSequence.descriptor
            peptideAbhprkDf = pd.DataFrame(peptideAbhprk)
            abhprkColumnsNum = [*range(1, len(peptideAbhprkDf.columns.tolist()) + 1)]
            abhprkColumnsStr = ['ABHPRK_'] * len(peptideAbhprkDf.columns.tolist())
            abhprkColumns = list(map(lambda x, y: x + '_' + str(y), abhprkColumnsStr, abhprkColumnsNum))
            peptideAbhprkDf.columns = abhprkColumns
            featureDf = pd.concat([featureDf, peptideAbhprkDf], axis=1)

        if self.featureDict.get("argos") is True:
            argosSequence = PeptideDescriptor(self.sequencelist, "argos")
            argosSequence.calculate_moment()
            peptideArgos = argosSequence.descriptor
            peptideArgosDf = pd.DataFrame(peptideArgos)
            featureDf.insert(len(featureDf.columns.tolist()), 'Argos', peptideArgosDf)

        if self.featureDict.get("bulkiness") is True:
            bulkinessSequence = PeptideDescriptor(self.sequencelist, "bulkiness")
            bulkinessSequence.calculate_moment()
            peptideBulkiness = bulkinessSequence.descriptor
            peptideBulkinessDf = pd.DataFrame(peptideBulkiness)
            featureDf.insert(len(featureDf.columns.tolist()), 'Bulkiness', peptideBulkinessDf)

        if self.featureDict.get("charge_phys") is True:
            physSequence = PeptideDescriptor(self.sequencelist, "charge_phys")
            physSequence.calculate_moment()
            peptideCharge_phys = physSequence.descriptor
            peptideCharge_physDf = pd.DataFrame(peptideCharge_phys)
            featureDf.insert(len(featureDf.columns.tolist()), 'Charge_phys', peptideCharge_physDf)

        if self.featureDict.get("charge_acid") is True:
            acidSequence = PeptideDescriptor(self.sequencelist, "charge_acid")
            acidSequence.calculate_moment()
            peptideCharge_acid = acidSequence.descriptor
            peptideCharge_acidDf = pd.DataFrame(peptideCharge_acid)
            featureDf.insert(len(featureDf.columns.tolist()), 'Charge_acid', peptideCharge_acidDf)

        if self.featureDict["cougar"][0] is True:
            cougarSequence = PeptideDescriptor(self.sequencelist, "cougar")
            cougarSequence.calculate_crosscorr(window=self.featureDict["cougar"][1])
            peptideCougar = cougarSequence.descriptor
            peptideCougarDf = pd.DataFrame(peptideCougar)
            cougarColumnsNum = [*range(1, len(peptideCougarDf.columns.tolist()) + 1)]
            cougarColumnsStr = ['Cougar_'] * len(peptideCougarDf.columns.tolist())
            cougarColumns = list(map(lambda x, y: x + '_' + str(y), cougarColumnsStr, cougarColumnsNum))
            peptideCougarDf.columns = cougarColumns
            featureDf = pd.concat([featureDf, peptideCougarDf], axis=1)

        if self.featureDict["ez"][0] is True:
            ezSequence = PeptideDescriptor(self.sequencelist, "Ez")
            ezSequence.calculate_crosscorr(window=self.featureDict["ez"][1])
            peptideEz = ezSequence.descriptor
            peptideEzDf = pd.DataFrame(peptideEz)
            ezColumnsNum = [*range(1, len(peptideEzDf.columns.tolist()) + 1)]
            ezColumnsStr = ['Ez_'] * len(peptideEzDf.columns.tolist())
            ezColumns = list(map(lambda x, y: x + '_' + str(y), ezColumnsStr, ezColumnsNum))
            peptideEzDf.columns = ezColumns
            featureDf = pd.concat([featureDf, peptideEzDf], axis=1)

        if self.featureDict.get("flexibility") is True:
            flexibilitySequence = PeptideDescriptor(self.sequencelist, "flexibility")
            flexibilitySequence.calculate_moment()
            peptideFlexibility = flexibilitySequence.descriptor
            peptideFlexibilityDf = pd.DataFrame(peptideFlexibility)
            featureDf.insert(len(featureDf.columns.tolist()), 'Flexibility', peptideFlexibilityDf)

        if self.featureDict.get("gravy") is True:
            gravySequence = PeptideDescriptor(self.sequencelist, "Gravy")
            gravySequence.calculate_moment()
            peptideGravy = gravySequence.descriptor
            peptideGravyDf = pd.DataFrame(peptideGravy)
            featureDf.insert(len(featureDf.columns.tolist()), 'Gravy', peptideGravyDf)

        if self.featureDict.get("levitt_alpha") is True:
            alphaSequence = PeptideDescriptor(self.sequencelist, "levitt_alpha")
            alphaSequence.calculate_moment()
            peptideLevitt_alpha = alphaSequence.descriptor
            peptideLevitt_alphaDf = pd.DataFrame(peptideLevitt_alpha)
            featureDf.insert(len(featureDf.columns.tolist()), 'Levitt_alpha', peptideLevitt_alphaDf)

        if self.featureDict.get("mss") is True:
            mssSequence = PeptideDescriptor(self.sequencelist, "MSS")
            mssSequence.calculate_moment()
            peptideMSS = mssSequence.descriptor
            peptideMSSDf = pd.DataFrame(peptideMSS)
            featureDf.insert(len(featureDf.columns.tolist()), 'MSS', peptideMSSDf)

        if self.featureDict["msw"][0] is True:
            mswSequence = PeptideDescriptor(self.sequencelist, "MSW")
            mswSequence.calculate_crosscorr(window=self.featureDict["msw"][1])
            peptideMSW = mswSequence.descriptor
            peptideMSWDf = pd.DataFrame(peptideMSW)
            mswColumnsNum = [*range(1, len(peptideMSWDf.columns.tolist()) + 1)]
            mswColumnsStr = ['MSW_'] * len(peptideMSWDf.columns.tolist())
            mswColumns = list(map(lambda x, y: x + '_' + str(y), mswColumnsStr, mswColumnsNum))
            peptideMSWDf.columns = mswColumns
            featureDf = pd.concat([featureDf, peptideMSWDf], axis=1)

        if self.featureDict.get("peparc") is True:
            pepArcSequence = PeptideDescriptor(self.sequencelist, "pepArc")
            pepArcSequence.calculate_arc(modality='max')
            peptidePepArc = pepArcSequence.descriptor
            peptidePepArcDf = pd.DataFrame(peptidePepArc)
            featureDf.insert(len(featureDf.columns.tolist()), 'pepArc', peptidePepArcDf)

        if self.featureDict.get("polarity") is True:
            polaritySequence = PeptideDescriptor(self.sequencelist, "polarity")
            polaritySequence.calculate_moment()
            peptidePolarity = polaritySequence.descriptor
            peptidePolarityDf = pd.DataFrame(peptidePolarity)
            featureDf.insert(len(featureDf.columns.tolist()), 'Polarity', peptidePolarityDf)

        if self.featureDict.get("refractivity") is True:
            refractivitySequence = PeptideDescriptor(self.sequencelist, "refractivity")
            refractivitySequence.calculate_moment()
            peptideRefractivity = refractivitySequence.descriptor
            peptideRefractivityDf = pd.DataFrame(peptideRefractivity)
            featureDf.insert(len(featureDf.columns.tolist()), 'Refractivity', peptideRefractivityDf)

        if self.featureDict.get("tm_tend") is True:
            tendSequence = PeptideDescriptor(self.sequencelist, "TM_tend")
            tendSequence.calculate_moment()
            peptideTM_tend = tendSequence.descriptor
            peptideTM_tendDf = pd.DataFrame(peptideTM_tend)
            featureDf.insert(len(featureDf.columns.tolist()), 'TM_tend', peptideTM_tendDf)

        if self.featureDict["z3"][0] is True:
            z3Sequence = PeptideDescriptor(self.sequencelist, "z3")
            z3Sequence.calculate_crosscorr(window=self.featureDict["z3"][1])
            peptideZ3 = z3Sequence.descriptor
            peptideZ3Df = pd.DataFrame(peptideZ3)
            z3ColumnsNum = [*range(1, len(peptideZ3Df.columns.tolist()) + 1)]
            z3ColumnsStr = ['Z3_'] * len(peptideZ3Df.columns.tolist())
            z3Columns = list(map(lambda x, y: x + '_' + str(y), z3ColumnsStr, z3ColumnsNum))
            peptideZ3Df.columns = z3Columns
            featureDf = pd.concat([featureDf, peptideZ3Df], axis=1)

        if self.featureDict["z5"][0] is True:
            z5Sequence = PeptideDescriptor(self.sequencelist, "z5")
            z5Sequence.calculate_crosscorr(window=self.featureDict["z5"][1])
            peptideZ5 = z5Sequence.descriptor
            peptideZ5Df = pd.DataFrame(peptideZ5)
            z5ColumnsNum = [*range(1, len(peptideZ5Df.columns.tolist()) + 1)]
            z5ColumnsStr = ['Z5_'] * len(peptideZ5Df.columns.tolist())
            z5Columns = list(map(lambda x, y: x + '_' + str(y), z5ColumnsStr, z5ColumnsNum))
            peptideZ5Df.columns = z5Columns
            featureDf = pd.concat([featureDf, peptideZ5Df], axis=1)

        if self.featureDict.get("formula") is True:
            formulaSequence = GlobalDescriptor(self.sequencelist)
            formulaSequence.formula()
            formulaKeyList = ["formula_C", "formula_H", "formula_N", "formula_O", "formula_S"]
            peptideFormula = []
            for sumResult in formulaSequence.descriptor:
                element_dict = {'C': 0, "H": 0, "N": 0, "O": 0, "S": 0}
                elements = sumResult[0].split()
                for element in elements:
                    symbol = element[0]
                    count = int(element[1:])
                    element_dict[symbol] = count
                peptideFormula.append(element_dict)
            formulaDf = pd.DataFrame(peptideFormula)
            formulaDf.columns = formulaKeyList
            featureDf = pd.concat([featureDf, formulaDf], axis=1)

        if self.featureDict.get("boman_index") is True:
            bomanSequence = GlobalDescriptor(self.sequencelist)
            bomanSequence.boman_index()
            peptideBoman_Index = bomanSequence.descriptor
            peptideBoman_IndexDf = pd.DataFrame(peptideBoman_Index)
            featureDf.insert(len(featureDf.columns.tolist()), 'Boman_Index', peptideBoman_IndexDf)

        if self.featureDict.get("eisenberg") is True:
            eisenbergSequence = PeptideDescriptor(self.sequencelist, "eisenberg")
            eisenbergSequence.calculate_moment()
            peptideEisenberg = eisenbergSequence.descriptor
            peptideEisenbergDf = pd.DataFrame(peptideEisenberg)
            featureDf.insert(len(featureDf.columns.tolist()), 'Eisenberg', peptideEisenbergDf)

        if self.featureDict.get("hopp_woods") is True:
            hopp_woodsSequence = PeptideDescriptor(self.sequencelist, "hopp-woods")
            hopp_woodsSequence.calculate_moment()
            peptideHopp_woods = hopp_woodsSequence.descriptor
            peptideHopp_woodsDf = pd.DataFrame(peptideHopp_woods)
            featureDf.insert(len(featureDf.columns.tolist()), 'Hopp_woods', peptideHopp_woodsDf)

        if self.featureDict.get("janin") is True:
            janinSequence = PeptideDescriptor(self.sequencelist, "janin")
            janinSequence.calculate_moment()
            peptideJanin = janinSequence.descriptor
            peptideJaninDf = pd.DataFrame(peptideJanin)
            featureDf.insert(len(featureDf.columns.tolist()), 'Janin', peptideJaninDf)

        if self.featureDict.get("kytedoolittle") is True:
            kytedoolittleSequence = PeptideDescriptor(self.sequencelist, "kytedoolittle")
            kytedoolittleSequence.calculate_moment()
            peptideKytedoolittle = kytedoolittleSequence.descriptor
            peptideKytedoolittleDf = pd.DataFrame(peptideKytedoolittle)
            featureDf.insert(len(featureDf.columns.tolist()), 'Kytedoolittle', peptideKytedoolittleDf)

        featureDf.set_index(0, drop=True, inplace=True)
        self.featureDf = featureDf

    def getOutputDf(self):
        return self.featureDf
