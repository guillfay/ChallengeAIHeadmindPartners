matching = [('M0538OCALM35R', 'image_1'), ('M1286ZRHMM918', 'image_10'), ('DIORCLAN2J5G1I', 'image_11'), ('M0446CSAZM49E', 'image_12'), ('30MTSUR14A0', 'image_13'), ('30MTSUMR41A0', 'image_14'), ('M0565OCEAM900', 'image_15'), ('M9242UTHRM928', 'image_16'), ('KDI717VEAS900', 'image_17'), ('KCK304CVES03W', 'image_18'), ('KCK278BCRS31W', 'image_19'), ('M0505SNEAM900', 'image_2'), ('KCK304CVES03W', 'image_20'), ('KCB637CRUS900', 'image_21'), ('KCK233TOKS56B', 'image_22'), ('KCK304CVES03W', 'image_23'), ('KCI736CQCS900', 'image_24'), ('30MTS3UXR12A1', 'image_25'), ('30MTS3UXR12A1', 'image_26'), ('ATTITUDE22M01I', 'image_27'), ('BOBYR1UXR42FR', 'image_28'), ('DSGTS1UXR10A0', 'image_29'), ('M0531PWRTM116', 'image_3'), ('BOBYS2UXR10A1', 'image_30'), ('DSGTS1UXR20B0', 'image_31'), ('BOBYR1UXR10A0', 'image_32'), ('DIORAMA8F0T41I', 'image_33'), ('30MTSUMR41A0', 'image_34'), ('CLUBM4UWR45A0', 'image_35'), ('M9203SBAVM989', 'image_36'), ('M9203SBAVM989', 'image_37'), ('M9203UTFQM928', 'image_38'), ('S2110UMOSM900', 'image_39'), ('S2110UMOSM991', 'image_4'), ('M0446CBAAM64H', 'image_40'), ('M0446ILLOM50P', 'image_41'), ('M1265ZTDTM16E', 'image_42'), ('S5555CRHMM918', 'image_43'), ('M9203IBAVM50P', 'image_44'), ('M505SOAHGM900', 'image_45'), ('M9000CHECM920', 'image_46'), ('M9242BWDEM090', 'image_47'), ('CD13416ZA0070000', 'image_48'), ('CD153BIZA0150000', 'image_49'), ('S2110UMOSM991', 'image_5'), ('S2057OBAEM41G', 'image_50'), ('111P38A6805X0861', 'image_51'), ('M0505OCALM20Y', 'image_52'), ('CAL44550M42R', 'image_53'), ('KCK327NVCS92Z', 'image_54'), ('KCK311VEAS900', 'image_55'), ('KCK325VEAS19O', 'image_56'), ('KDB717ACAS44W', 'image_57'), ('KCQ369VEAS900', 'image_58'), ('KDB578TJES36W', 'image_59'), ('KDB704LNYS26U', 'image_6'), ('KDP867MPPS19W', 'image_60'), ('KDP967PPWS17X', 'image_61'), ('KDI611SCNS52X', 'image_62'), ('KCC201TFLS33R', 'image_63'), ('M9220UPGOM900', 'image_64'), ('KDI607VSOS900', 'image_65'), ('KCI491CFMS900', 'image_66'), ('KCI717VEAS900', 'image_67'), ('KCI675SCRS52X', 'image_68'), ('KCK304CVES03W', 'image_69'), ('S2110UMOSM900', 'image_7'), ('KCK177CVAS12X', 'image_70'), ('KCK211DAMS17X', 'image_71'), ('KCK318CMPS10W', 'image_72'), ('30MTSUR14A0', 'image_73'), ('ATTITUDE22M01I', 'image_74'), ('CLUBM4UWR45A0', 'image_75'), ('DSGTS1UXR10A0', 'image_76'), ('DSGTS1UXR10A0', 'image_77'), ('CLUBM4UWR45A0', 'image_78'), ('S5652CCEHM900', 'image_79'), ('M0531OALSM659', 'image_8'), ('S5652CCEHM900', 'image_80'), ('M1286ZRHMM918', 'image_9')]
import pandas as pd
import ast

df=pd.DataFrame(matching)
# df.rename(columns={0:"DAM", 1: "test"})
# df.set_index("test")
# df.to_csv("matching.csv")

matching_df = pd.read_csv("matching.csv", sep=';')
# Fonction pour convertir les chaînes en listes
def process_dam(value):
    try:
        # Si la valeur commence par un crochet, utiliser ast.literal_eval
        if value.startswith('['):
            return ast.literal_eval(value)
        # Si la valeur contient des virgules, diviser par les virgules
        elif ',' in value:
            return [v.strip() for v in value.split(',')]
        else:
            # Sinon, encapsuler la valeur dans une liste
            return [value]
    except Exception as e:
        print(f"Erreur lors de l'évaluation de la valeur : {value}. Erreur : {e}")
        return [value]

# Appliquer la fonction à la colonne 'Dam'
matching_df['DAM'] = matching_df['DAM'].apply(process_dam)
print(matching_df['DAM'].to_list())

