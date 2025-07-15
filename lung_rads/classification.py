import pandas as pd

from lungrads_mod import nodule
from lungrads_mod import lung_rads_classifier


def determine_attenuation(row):
    if row['O nódulo é sólido ou em partes moles?']:
        return 'Sólido'
    elif row['O nódulo tem densidade semissólida ou parcialmente sólida?']:
        return 'Parcialmente Sólido'
    elif row['O nódulo é em vidro fosco?']:
        return 'Vidro Fosco'
    return 'Desconhecido'

# Função para determinar os bordos (espiculado ou não)
def determine_edges(row):
    return 'Espiculada' if row['O nódulo é espiculado, irregular ou mal definido?'] else 'Não Espiculada'

# Função para classificar os nódulos
def classify_nodules(df):
    nodules_classification = []
    for _, row in df.iterrows():
        attenuation = determine_attenuation(row)
        edges = determine_edges(row)
        calcification = row['O nódulo é calcificado?']
        location = row['Localização do nódulo']
        if row['Tamanho do nódulo (mm)'] != "False":
            diameter = float(row['Tamanho do nódulo (mm)'])
        
        # Criar instância de Nodule
        single_nodule = nodule.Nodule(attenuation=attenuation, edges=edges, calcification=calcification, localization=location, size=diameter, solid_component_size=diameter)
        
        # Classificar usando LungRADSClassifier
        classifier = lung_rads_classifier.LungRADSClassifier(single_nodule)
        results = classifier.classifier()
        #print(results)
        #print("\n================================")
        
        # Adicionar resultado à lista
        nodules_classification.append(results)
    
    return nodules_classification
