from django.http import HttpResponse
from django.template import loader
from django.shortcuts import render

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


#Nettoyer les données 
data2018 = pd.read_csv("donnees\\valeursfoncieres-2018.txt",sep="|")
data2022 = pd.read_csv("donnees\\valeursfoncieres-2022.txt",sep="|")
for col in data2018.columns :
    if data2018[col].isnull().sum()>= 3000000 :
        data2018.drop(columns=[col],inplace=True)
for col in data2022.columns :
    if data2022[col].isnull().sum()>=3000000:
        data2022.drop(columns=[col],inplace=True)
data2022 = data2022.drop_duplicates()
data2018 = data2018.drop_duplicates()

#Garder que 50%
import numpy as np
from sklearn.model_selection import train_test_split
data2018mini, rest =train_test_split(data2018, test_size=0.5, random_state=42)    
data2022mini, rest2 =train_test_split(data2022, test_size=0.5, random_state=42)

data2022mini["Valeur fonciere"]= data2022mini["Valeur fonciere"].str.replace(",",".")
data2022mini["Valeur fonciere"]= data2022mini["Valeur fonciere"].astype(float)
data2022mini["Valeur fonciere"].head(20)
data2018mini["Valeur fonciere"]= data2018mini["Valeur fonciere"].str.replace(",",".")
data2018mini["Valeur fonciere"]= data2018mini["Valeur fonciere"].astype(float)
data2018mini["Valeur fonciere"].head(20)
# Replace missing values with -1
data2018mini["Code postal"] = data2018mini["Code postal"].fillna(-1)
data2022mini["Code postal"] = data2022mini["Code postal"].fillna(-1)
# Convert the column to integer
data2018mini["Code postal"] = data2018mini["Code postal"].astype(int)
data2022mini["Code postal"] = data2022mini["Code postal"].astype(int)
department_codes = data2022mini['Code departement'].astype(str)
# Remove the decimal places from the department codes
data2022mini['Code departement'] = department_codes.str.split('.', expand=True)[0]    
department_codes2 = data2018mini['Code departement'].astype(str)
# Remove the decimal places from the department codes
data2018mini['Code departement'] = department_codes.str.split('.', expand=True)[0]

data2022mini.drop(data2022mini.index[data2022mini.iloc[:,3].isnull()],0, inplace=True)
data2018mini.drop(data2018mini.index[data2018mini.iloc[:,3].isnull()],0, inplace=True)

# On retire les 5% les plus grands (aberrants)
seuil = data2022mini["Valeur fonciere"].quantile(0.95)
data2022miniF = data2022mini[data2022mini["Valeur fonciere"] < seuil]
seuil = data2018mini["Valeur fonciere"].quantile(0.95)
data2018miniF = data2018mini[data2018mini["Valeur fonciere"] < seuil]


def index3(request):
    template = loader.get_template("template1.html")
    

    if(request.GET['model']=="visu1"):
        plot_html=visu1()
    elif(request.GET['model']=="visu2"):
        plot_html=visu2()
    elif(request.GET['model']=="visu3"):
        plot_html=visu3()
    elif(request.GET['model']=="visu4"):
        plot_html=visu4()
    elif(request.GET['model']=="visu5"):
        plot_html=visu5()
    elif(request.GET['model']=="visu6"):
        plot_html=visu6()
    elif(request.GET['model']=="visu7"):
        plot_html=visu7()
    elif(request.GET['model']=="visu8"):
        plot_html=visu8()
    elif(request.GET['model']=="visu9"):
        plot_html=visu9()
    elif(request.GET['model']=="visu10"):
        plot_html=visu10()
    elif(request.GET['model']=="visu11"):
        plot_html=visu11()
    elif(request.GET['model']=="visu12"):
        plot_html=visu12()
    elif(request.GET['model']=="visu13"):
        plot_html=visu13()
    elif(request.GET['model']=="visu14"):
        plot_html=visu14()
    elif(request.GET['model']=="visu15"):
        plot_html=visu15()
    elif(request.GET['model']=="visu16"):
        plot_html=visu16()
    elif(request.GET['model']=="visu17"):
        plot_html=visu17()
    elif(request.GET['model']=="visu18"):
        plot_html=visu18()
    elif(request.GET['model']=="visu19"):
        plot_html=visu19()
    elif(request.GET['model']=="visu20"):
        plot_html=visu20()
    elif(request.GET['model']=="visu21"):
       plot_html=visu21()
    elif(request.GET['model']=="visu22"):
        plot_html=visu22()
    elif(request.GET['model']=="visu23"):
        plot_html=visu23()
    elif(request.GET['model']=="visu24"):
        plot_html=visu24()
    elif(request.GET['model']=="visu25"):
        plot_html=visu25()
    elif(request.GET['model']=="visu26"):
        plot_html=visu26()
    elif(request.GET['model']=="visu27"):
        plot_html=visu27()
    elif(request.GET['model']=="visu28"):
        plot_html=visu28()
    elif(request.GET['model']=="visu29"):
        plot_html=visu29()
    else:
        plot_html=visu1()
    

    context = {
        'plot_html': plot_html,
    }
    
    return HttpResponse(template.render(context, request))

    template = loader.get_template("template1.html")
    

    if(request.GET['model']=="visu1"):
        plot_html=visu1()
    elif(request.GET['model']=="visu2"):
        plot_html=visu2()
    elif(request.GET['model']=="visu3"):
        plot_html=visu3()
    elif(request.GET['model']=="visu4"):
        plot_html=visu4()
    elif(request.GET['model']=="visu5"):
        plot_html=visu5()
    elif(request.GET['model']=="visu6"):
        plot_html=visu6()
    elif(request.GET['model']=="visu7"):
        plot_html=visu7()
    elif(request.GET['model']=="visu8"):
        plot_html=visu8()
    elif(request.GET['model']=="visu9"):
        plot_html=visu9()
    elif(request.GET['model']=="visu10"):
        plot_html=visu10()
    elif(request.GET['model']=="visu11"):
        plot_html=visu11()
    elif(request.GET['model']=="visu12"):
        plot_html=visu12()
    elif(request.GET['model']=="visu13"):
        plot_html=visu13()
    elif(request.GET['model']=="visu14"):
        plot_html=visu14()
    elif(request.GET['model']=="visu15"):
        plot_html=visu15()
    elif(request.GET['model']=="visu16"):
        plot_html=visu16()
    elif(request.GET['model']=="visu17"):
        plot_html=visu17()
    elif(request.GET['model']=="visu18"):
        plot_html=visu18()
    elif(request.GET['model']=="visu19"):
        plot_html=visu19()
    elif(request.GET['model']=="visu20"):
        plot_html=visu20()
    elif(request.GET['model']=="visu21"):
       plot_html=visu21()
    elif(request.GET['model']=="visu22"):
        plot_html=visu22()
    elif(request.GET['model']=="visu23"):
        plot_html=visu23()
    elif(request.GET['model']=="visu24"):
        plot_html=visu24()
    elif(request.GET['model']=="visu25"):
        plot_html=visu25()
    elif(request.GET['model']=="visu26"):
        plot_html=visu26()
    elif(request.GET['model']=="visu27"):
        plot_html=visu27()
    elif(request.GET['model']=="visu28"):
        plot_html=visu28()
    elif(request.GET['model']=="visu29"):
        plot_html=visu29()
    else:
        plot_html=Firstgraph()
    

    context = {
        'plot_html': plot_html,
    }
    
    return HttpResponse(template.render(context, request))

def index(request):
    
    #return HttpResponse(template.render(context, request))
    return render(request, "template1.html")
    #return HttpResponse(template.render(context, request))
    
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import plotly.express as px


import pandas as pd
import mpld3


def visu1():
    plt.hist(data2022miniF["Valeur fonciere"], bins=5)
    plt.xlabel("Valeur fonciere")
    plt.ylabel("Nombre de transactions")

    # Convert the plot to HTML
    plot_html = mpld3.fig_to_html(plt.gcf())
    
    return plot_html

def visu2():
    sns.kdeplot(data2022miniF["Valeur fonciere"])
    plt.xlabel("Valeur fonciere")
    plt.ylabel("Densité")
    plot_html2 = mpld3.fig_to_html(plt.gcf())
    
    return plot_html2

def visu3():
    #Boxplot
    plt.boxplot(data2022miniF["Valeur fonciere"])
    plt.ylabel("Valeur fonciere")
    plot_html3 = mpld3.fig_to_html(plt.gcf())
    
    return plot_html3

def visu4():
    types_locaux = data2022miniF["Type local"].value_counts()

    plt.figure(figsize=(15,5))
    plt.pie(types_locaux.values, labels=types_locaux.index, autopct='%1.1f%%')
    plt.legend(fontsize=14)
    plt.axis('equal')
    plot_html = mpld3.fig_to_html(plt.gcf())
    
    return plot_html

def visu5():
    data2022miniF = data2022mini[data2022mini["Surface terrain"] < 100000]
    data2022miniF = data2022miniF[data2022miniF["Surface reelle bati"] < 1000]
    
    plt.scatter(data2022miniF["Surface terrain"], data2022miniF["Surface reelle bati"])
    plt.xlabel("Surface terrain")
    plt.ylabel("Surface reelle bati")
    
    plot_html = mpld3.fig_to_html(plt.gcf())
    
    return plot_html

def visu6():
    data2022miniF = data2022mini[data2022mini["Surface terrain"] < 100000]
    data2022miniF = data2022miniF[data2022miniF["Surface reelle bati"] < 1000]
    sns.relplot(x=data2022miniF["Surface terrain"],y=data2022miniF["Surface reelle bati"],hue=data2022miniF["Type local"],data=data2022miniF)
    plot_html = mpld3.fig_to_html(plt.gcf())
    
    return plot_html

def visu7():
    #Boxplot surface batie/departement
    
    df_surfacebati = data2022mini[["Code departement", "Surface reelle bati"]]
    # Calculate the quartiles and IQR
    Q1 = df_surfacebati["Surface reelle bati"].quantile(0.25)
    Q3 = df_surfacebati["Surface reelle bati"].quantile(0.75)
    IQR = Q3 - Q1
    
    # Filter out outliers
    df_surfacebati = df_surfacebati[(df_surfacebati["Surface reelle bati"] >= Q1 - 1.5 * IQR) & (df_surfacebati["Surface reelle bati"] <= Q3 + 1.5 * IQR)]
    # Groupby pour regrouper les superficies par ville
    grouped = df_surfacebati.groupby("Code departement")
    # Créer une liste de DataFrames pour chaque ville
    dfsb = [grouped.get_group(x) for x in grouped.groups]  
    # Créer une liste de noms de villes
    names = [x for x in grouped.groups]
    # Créer le boxplot
    plt.figure(figsize=(15,8))
    plt.boxplot([x["Surface reelle bati"] for x in dfsb], labels=names)
    ax = plt.gca()
    ax.set_xticklabels(names, rotation=90, fontsize=8)
    plt.title("Distribution des superficies des biens immobiliers par départements")
    plt.xlabel("Départements")
    plt.ylabel("Superficie (m²)")
    plt.ylim(0, 300)
    plt.xlim(0, 100)
    plot_html = mpld3.fig_to_html(plt.gcf())
    
    return plot_html

def visu8():
    #Histogramme
    df = data2022mini
    # Grouper les données par département et calculer la surface réelle bâtie moyenne
    average_surface_batie = df.groupby('Code departement')['Surface reelle bati'].mean()
    # Création du graphique à barres
    plt.figure(figsize=(17, 8))
    average_surface_batie.plot(kind='bar', color='blue')
    # Ajout des labels et du titre
    plt.xlabel('Département')
    plt.ylabel('Surface réelle bâtie moyenne')
    plt.title('Surface réelle bâtie moyenne par département')
    plot_html = mpld3.fig_to_html(plt.gcf())
    
    return plot_html

def visu9():
    #Histogramme empilé 
    # Créer un pivot table pour avoir la somme des surfaces par ville et type de propriété
    table = pd.pivot_table(data2022mini, values='Surface reelle bati', index='Code departement', columns='Type local', aggfunc=sum)
    # Créer l'histogramme empilé
    table.plot(kind='bar', stacked=True, figsize=(15, 8))
    # Ajouter les titres et les labels
    plt.title("Distribution des superficies des biens immobiliers par département et par type de propriété")
    plt.xlabel("Département")
    plt.ylabel("Surface totale (m2)")
    plot_html = mpld3.fig_to_html(plt.gcf())
    
    return plot_html


def visu10():
    maison_counts = data2022mini[data2022mini['Type local'] == 'Maison']['Code departement'].astype(str).value_counts().sort_index()

    x_values = np.arange(len(maison_counts))
    y_values = maison_counts.values
    tick_labels = maison_counts.index
    
    fig, ax = plt.subplots(figsize=(15, 6)) # increase the size of the figure
    ax.bar(x_values, y_values, alpha=0.5, color='red', label='2022', width=0.8)
    ax.set_xticks(x_values)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=8)
    ax.set_title("Distribution du nombre de maison")
    ax.set_xlabel("Départements")
    ax.set_ylabel("Nombre de Transactions")
    ax.legend()
    plot_html = mpld3.fig_to_html(plt.gcf())
    
    return plot_html


def visu11():
    """ Read in the GeoJSON file for the department borders
    france_map = gpd.read_file("https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-version-simplifiee.geojson")
    # Calculate the count of 'Maison' by department and sort in descending order
    maison_counts = data2022mini[data2022mini['Type local'] == 'Maison']['Code departement'].astype(str).value_counts().sort_index()
    # Calculate the 90th percentile of the "Maison" values
    q90 = maison_counts.quantile(0.8)
    # Assign a color gradient to the departments based on the count of 'Maison'
    france_map = france_map.rename(columns={'code': 'Code departement'})
    france_map['color'] = france_map['Code departement'].apply(lambda x: cm.get_cmap('magma')(maison_counts.get(x, 0)/q90))
    # Plot the map with the colored departments
    fig, ax = plt.subplots(figsize=(12, 8))
    france_map.plot(ax=ax, color=france_map['color'], edgecolor='white', linewidth=0.5)
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cm.get_cmap('magma'), norm=plt.Normalize(vmin=maison_counts.min(), vmax=maison_counts.max()))
    sm._A = []
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.ax.set_ylabel('Nombre de maisons', rotation=270, labelpad=20)
    ax.set_title("Nombre d'opérations (Vente, echange, ...) avec des maisons par département en 2022")
    ax.axis('off')
    plot_html = mpld3.fig_to_html(plt.gcf())
        
    return plot_html
"""
    return 0

def visu12():
    df = data2022mini
    # Grouper les données par département et type local
    # Convert "Code departement" column to string data type
    df["Code departement"] = df["Code departement"].astype(str)
    # Sélection des départements 75, 92, 27 et 50
    selected_departments = ['27', '50', '75', '92']
    df_selected = df[df["Code departement"].isin(selected_departments)]
    # Grouper les données par département et type local
    grouped_data = df_selected.groupby(['Code departement', 'Type local']).size().unstack()
    # Création du graphique à secteurs pour chaque département
    for departement, row in grouped_data.iterrows():
        labels = row.index
        sizes = row.values
        plt.figure()
        plt.pie(sizes, labels=labels, autopct='%1.1f%%')
        plt.title(f"Répartition des types locaux dans le département {departement}")
        plt.axis('equal')
    plot_html = mpld3.fig_to_html(plt.gcf())
    
    return plot_html


def visu13():
    # Calculate the counts for each property type
    maison_counts = data2022mini[data2022mini['Type local'] == 'Maison']['Code departement'].astype(str).value_counts().sort_index()
    appartement_counts = data2022mini[data2022mini['Type local'] == 'Appartement']['Code departement'].astype(str).value_counts().sort_index()
    local_counts = data2022mini[data2022mini['Type local'] == 'Local industriel. commercial ou assimilé']['Code departement'].astype(str).value_counts().sort_index()
    dependance_counts = data2022mini[data2022mini['Type local'] == 'Dépendance']['Code departement'].astype(str).value_counts().sort_index()
    # Combine the counts into a single DataFrame
    counts_df = pd.DataFrame({'Appartement': appartement_counts,
                              'Dépendance': dependance_counts,
                              'Local': local_counts,
                              'Maison': maison_counts})
    # Create the stacked histogram
    counts_df.plot(kind='bar', stacked=True, figsize=(15, 8))
    # Add titles and labels
    plt.title("Distribution des nombres de biens immobiliers par département et par type de propriété")
    plt.xlabel("Département")
    plt.ylabel("Nombre total")
    plot_html = mpld3.fig_to_html(plt.gcf())
    
    return plot_html


def visu14():
    locaux_counts = data2022mini[data2022mini['Type local'] == 'Local industriel. commercial ou assimilé']['Code departement'].astype(str).value_counts().sort_index()
    x_values = np.arange(len(locaux_counts))
    y_values = locaux_counts.values
    tick_labels = locaux_counts.index
    fig, ax = plt.subplots(figsize=(15, 6)) # increase the size of the figure
    ax.bar(x_values, y_values, alpha=0.5, color='green', label='2022', width=0.8)
    ax.set_xticks(x_values)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=8)
    ax.set_title("Distribution du nombre de locaux")
    ax.set_xlabel("Départements")
    ax.set_ylabel("Nombre de Transactions")
    ax.legend()
    plot_html = mpld3.fig_to_html(plt.gcf())
    
    return plot_html


def visu15():
    """
    # Read in the GeoJSON file for the department borders
    france_map = gpd.read_file("https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-version-simplifiee.geojson")
    
    # Calculate the count of 'Maison' by department and sort in descending order
    locaux_counts = data2022mini[data2022mini['Type local'] == 'Local industriel. commercial ou assimilé']['Code departement'].astype(str).value_counts().sort_index()
    
    # Calculate the 90th percentile of the "Maison" values
    q90 = locaux_counts.quantile(0.8)
    
    # Assign a color gradient to the departments based on the count of 'Maison'
    france_map = france_map.rename(columns={'code': 'Code departement'})
    france_map['color'] = france_map['Code departement'].apply(lambda x: cm.get_cmap('magma')(locaux_counts.get(x, 0)/q90))
    
    # Plot the map with the colored departments
    fig, ax = plt.subplots(figsize=(12, 8))
    france_map.plot(ax=ax, color=france_map['color'], edgecolor='white', linewidth=0.5)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cm.get_cmap('magma'), norm=plt.Normalize(vmin=locaux_counts.min(), vmax=locaux_counts.max()))
    sm._A = []
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.ax.set_ylabel('Nombre de locaux', rotation=270, labelpad=20)
    
    ax.set_title("Nombre d'opérations (Vente, echange, ...) avec des locaux par département en 2022")
    ax.axis('off')
    plt.show()
    plt.savefig('testGraph1/static/figure14.png')  # Save the figure to a file
    
    """
    #plot_html = mpld3.fig_to_html(plt.gcf())
    
    return 0


def visu16():
    """
    # Read in the data
    france_map = gpd.read_file("https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-version-simplifiee.geojson")
    france_map2 = gpd.read_file("https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-version-simplifiee.geojson")
    
    # Filter the data for 'Maison' type local (second plot)
    maison_data_2018 = data2018mini[data2018mini['Type local'] == 'Maison']
    diff_2018 = maison_data_2018.groupby(['Code departement'])[['Surface terrain', 'Surface reelle bati']].sum()
    diff_2018['diff'] = diff_2018['Surface reelle bati'] / diff_2018['Surface terrain']
    
    maison_data_2022 = data2022mini[data2022mini['Type local'] == 'Maison']
    diff_2022 = maison_data_2022.groupby(['Code departement'])[['Surface terrain', 'Surface reelle bati']].sum()
    diff_2022['diff'] = diff_2022['Surface reelle bati'] / diff_2022['Surface terrain']
    
    # Merge the department data with the difference data (second plot)
    diff_2018 = diff_2018.reset_index().rename(columns={'Code departement': 'code'})
    france_map['color_2018'] = france_map['code'].apply(lambda x: cm.get_cmap('magma')(diff_2018.get(x, 0)))
    
    diff_2022 = diff_2022.reset_index().rename(columns={'Code departement': 'code'})
    france_map2['color_2022'] = france_map2['code'].apply(lambda x: cm.get_cmap('magma')(diff_2022.get(x, 0)))
    
    # Create the choropleth map for the first plot
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot
    france_map.plot(cmap='magma', linewidth=0.5, edgecolor='white', legend=True, ax=ax[0])
    ax[0].set_title("Ratio de la taille des jardins (2018)")
    ax[0].axis('off')
    france_map2.plot(cmap='magma', linewidth=0.5, edgecolor='white', legend=True, ax=ax[1])
    ax[1].set_title("Ratio de la taille des jardins (2022)")
    ax[1].axis('off')
    
    
    # Add colorbar
    vminT = min(diff_2018['diff'].min(), diff_2022['diff'].min())
    vmaxT = max(diff_2018['diff'].max(), diff_2022['diff'].max())
    
    sm_2018 = plt.cm.ScalarMappable(cmap='magma', norm=plt.Normalize(vmin=vminT, vmax=vmaxT))
    sm_2018._A = []  # Need to override the fake data
    cbar_2018 = fig.colorbar(sm_2018, ax=ax[0], fraction=0.02, pad=0.04)
    cbar_2018.ax.set_ylabel("Ratio entre les surfaces batties de maison et la surface des terrains (2018)", rotation=270, labelpad=20)
    sm_2022 = plt.cm.ScalarMappable(cmap='magma', norm=plt.Normalize(vmin=vminT, vmax=vmaxT))
    sm_2022._A = []  # Need to override the fake data
    cbar_2022 = fig.colorbar(sm_2022, ax=ax[1], fraction=0.02, pad=0.04)
    cbar_2022.ax.set_ylabel("Ratio entre les surfaces batties de maison et la surface des terrains (2022)", rotation=270, labelpad=20)
    
    
    # Adjust the spacing between the plots
    plt.subplots_adjust(wspace=0.1)
    
    # Display the plot
    plt.show()
    plt.savefig('testGraph1/static/figure15.png')  # Save the figure to a file
    context1= {
            'figure_path': 'testGraph1/static/figure15.png'  # Pass the path to the template
            }

    
    """
    #plot_html = mpld3.fig_to_html(plt.gcf())
    
    return 0


def visu17():
    communes = data2022miniF["Commune"].value_counts().head(20)
    plt.figure(figsize=(20, 6))
    plt.bar(communes.index, communes.values)
    plt.xlabel("Commune")
    plt.ylabel("Nombre de transactions")
    plt.xticks(rotation=60)
    plot_html = mpld3.fig_to_html(plt.gcf())
    
    return plot_html

def visu18():
    pieces_par_ville = data2022miniF.groupby("Commune")["Nombre pieces principales"].mean()
    # Trier les villes par ordre décroissant en fonction du nombre moyen de pièces
    pieces_par_ville = pieces_par_ville.sort_values(ascending=False).head(15)
    plt.figure(figsize= (20,5))
    # Créer le graphique à barres
    plt.bar(pieces_par_ville.index, pieces_par_ville.values)
    # Ajouter des étiquettes d'axe et un titre
    plt.xlabel("Commune")
    plt.ylabel("Nombre moyen de pièces")
    plt.title("Nombre moyen de pièces par ville")
    # Faire pivoter les étiquettes des villes pour une meilleure lisibilité
    plt.xticks(rotation=60)
    plot_html = mpld3.fig_to_html(plt.gcf())
    
    return plot_html

def visu19():
    # Grouper les données par commune et calculer la surface moyenne du terrain
    terrain_par_commune = data2022miniF.groupby("Commune")["Surface reelle bati"].mean()
    # Trier les communes par ordre décroissant en fonction de la surface moyenne du terrain
    terrain_par_commune = terrain_par_commune.sort_values(ascending=False).head(15)
    plt.figure(figsize= (20,5))
    # Créer le graphique à barres
    plt.bar(terrain_par_commune.index, terrain_par_commune.values)
    # Ajouter des étiquettes d'axe et un titre
    plt.xlabel("Commune")
    plt.ylabel("Surface moyenne du terrain (m²)")
    plt.title("Communes où les construction sont les plus grandes ")
    # Faire pivoter les étiquettes des communes pour une meilleure lisibilité
    plt.xticks(rotation=60)
    plot_html = mpld3.fig_to_html(plt.gcf())
    
    return plot_html

def visu20():
    terrainB_par_commune = data2022miniF.groupby("Commune")["Surface terrain"].mean()
    # Trier les communes par ordre décroissant en fonction de la surface moyenne du terrain
    terrainB_par_commune = terrainB_par_commune.sort_values(ascending=False).head(15)
    plt.figure(figsize= (20,5))
    # Créer le graphique à barres
    plt.bar(terrainB_par_commune.index, terrainB_par_commune.values)
    # Ajouter des étiquettes d'axe et un titre
    plt.xlabel("Commune")
    plt.ylabel("Surface moyenne du terrain (m²)")
    plt.title("Communes où les terrains sont les plus grands")
    # Faire pivoter les étiquettes des communes pour une meilleure lisibilité
    plt.xticks(rotation=60)
    plot_html = mpld3.fig_to_html(plt.gcf())
    
    return plot_html

def visu21():
    # Grouper les données par ville et calculer le prix moyen par mètre carré et le nombre moyen de pièces principales
    prix_par_m2_par_ville = data2022miniF.groupby("Commune")["Valeur fonciere", "Surface reelle bati", "Nombre pieces principales"].mean()
    prix_par_m2_par_ville["Prix par m2"] = prix_par_m2_par_ville["Valeur fonciere"] / prix_par_m2_par_ville["Surface reelle bati"]
    # Ajouter une colonne avec le nom de la ville pour chaque ligne
    prix_par_m2_par_ville["Nom de la ville"] = prix_par_m2_par_ville.index
    plt.figure(figsize=(15,5))
    # Créer le graphique de dispersion
    plt.scatter(prix_par_m2_par_ville["Nombre pieces principales"], prix_par_m2_par_ville["Prix par m2"])
    # Ajouter le nom de chaque ville à côté de son point correspondant dans le graphique
    prix_par_m2_par_ville.apply(lambda row: plt.text(row["Nombre pieces principales"] + 0.1,
                                                      row["Prix par m2"] + 0.1,
                                                      row["Nom de la ville"]), axis=1)
    # Ajouter des étiquettes d'axe et un titre
    plt.xlabel("Nombre de pièces principales")
    plt.ylabel("Prix moyen par m²")
    plt.title("Relation entre le prix moyen par m² et le nombre de pièces principales par ville")
    plot_html = mpld3.fig_to_html(plt.gcf())
    
    return plot_html

def visu22():
    # Calculer le prix moyen par mètre carré pour chaque ville
    mean_prices = data2022miniF.groupby("Commune")["Valeur fonciere", "Surface reelle bati"].sum()
    mean_prices["Prix moyen par m2"] = mean_prices["Valeur fonciere"] / mean_prices["Surface reelle bati"]
    # Trier les villes par prix moyen croissant et décroissant
    lowest_prices = mean_prices.sort_values("Prix moyen par m2").head(10)
    lowest_prices = lowest_prices[4:9]
    #print(lowest_prices)
    #print(highest_prices)
    # Créer le graphique à barres
    fig, ax = plt.subplots(figsize=(15,5))
    ax.bar(lowest_prices.index, lowest_prices["Prix moyen par m2"], label="Villes avec les prix moyens les plus bas")
    # Ajouter la valeur du prix moyen sur chaque barre
    for i, price in enumerate(lowest_prices["Prix moyen par m2"]):
        ax.text(i, price, f"{price:.2f}", ha="center", va="bottom")
    # Ajouter une légende
    ax.legend()
    # Ajouter des étiquettes d'axe et un titre
    plt.xlabel("Villes")
    plt.ylabel("Prix moyen par m²")
    plt.title("Les 5 villes où le prix moyen du mètre carré est le plus bas")
    plot_html = mpld3.fig_to_html(plt.gcf())
    
    return plot_html

def visu23():
    mean_prices = data2022miniF.groupby("Commune")["Valeur fonciere", "Surface reelle bati"].sum()
    mean_prices["Prix moyen par m2"] = mean_prices["Valeur fonciere"] / mean_prices["Surface reelle bati"]
    #Calcul des prix moyen par m2 les plus élevés
    highest_prices = mean_prices.sort_values("Prix moyen par m2", ascending=False)
    # Remplacer les valeurs infinies par NaN
    highest_prices.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Supprimer les lignes contenant des valeurs NaN
    highest_prices.dropna(inplace=True)
    highest_prices=highest_prices.head(5)
    #création du graphique
    fig, ax = plt.subplots(figsize=(15,5))
    ax.bar(highest_prices.index, highest_prices["Prix moyen par m2"], label="Villes avec les prix moyens les plus élevés",color='red')
    # Ajouter la valeur du prix moyen sur chaque barre
    for i, price in enumerate(highest_prices["Prix moyen par m2"]):
        ax.text(i, price, f"{price:.2f}", ha="center", va="bottom")
    # Ajouter une légende
    ax.legend()
    # Ajouter des étiquettes d'axe et un titre
    plt.xlabel("Villes")
    plt.ylabel("Prix moyen par m²")
    plt.title("Les 5 villes où le prix moyen du mètre carré est le plus élevé")
    plot_html = mpld3.fig_to_html(plt.gcf())
    
    return plot_html

def visu24():
    df = data2022mini
    # Création du graphique en nuage de points
    plt.figure(figsize=(12, 8))
    for departement, data in df.groupby('Code departement'):
        plt.scatter(data['Surface reelle bati'], data['Valeur fonciere'], label=departement)
    # Ajout des labels et du titre
    plt.xlabel('Surface réelle bâtie')
    plt.ylabel('Valeur foncière')
    plt.title('Relation entre la valeur foncière et la surface réelle bâtie')
    
    plot_html = mpld3.fig_to_html(plt.gcf())
    
    return plot_html

def visu25():
    df = data2022mini
    # Définir le seuil pour la valeur foncière
    seuil_valeur_fonciere = 50000000  # Modifier le seuil selon vos besoins
    # Filtrer les données pour ne conserver que les points avec une valeur foncière supérieure au seuil
    df_filtre = df[df['Valeur fonciere'] > seuil_valeur_fonciere]
    # Création du graphique en nuage de points avec les valeurs extrêmes
    plt.figure(figsize=(12, 8))
    plt.scatter(df_filtre['Surface reelle bati'], df_filtre['Valeur fonciere'])
    # Ajout des annotations pour chaque point avec le numéro du département
    for i in range(len(df_filtre)):
        plt.annotate(df_filtre['Code departement'].iloc[i], (df_filtre['Surface reelle bati'].iloc[i], df_filtre['Valeur fonciere'].iloc[i]))
    # Ajout des labels et du titre
    plt.xlabel('Surface réelle bâtie')
    plt.ylabel('Valeur foncière')
    plt.title('Relation entre la valeur foncière et la surface réelle bâtie (valeurs extrêmes)')
    plot_html = mpld3.fig_to_html(plt.gcf())
    
    return plot_html

def visu26():
    df = data2022mini
    # Définir le seuil pour la surface réelle bâtie
    seuil_surface_batie = 4000  # Modifier le seuil selon vos besoins
    # Filtrer les données pour ne conserver que les points avec une surface réelle bâtie supérieure au seuil
    df_filtre = df[df['Surface reelle bati'] > seuil_surface_batie]
    # Création du graphique en nuage de points avec les valeurs extrêmes
    plt.figure(figsize=(12, 8))
    plt.scatter(df_filtre['Surface reelle bati'], df_filtre['Valeur fonciere'])
    # Ajout des annotations pour chaque point avec le numéro du département
    for i in range(len(df_filtre)):
        plt.annotate(df_filtre['Code departement'].iloc[i], (df_filtre['Surface reelle bati'].iloc[i], df_filtre['Valeur fonciere'].iloc[i]))
    # Ajout des labels et du titre
    plt.xlabel('Surface réelle bâtie')
    plt.ylabel('Valeur foncière')
    plt.title('Relation entre la valeur foncière et la surface réelle bâtie (valeurs extrêmes)')
    plot_html = mpld3.fig_to_html(plt.gcf())
    
    return plot_html


def visu27():
    df2022 = data2022mini
    df2018 = data2018mini
    # Conversion des données de type "object" en données de type numérique pour la colonne "Code departement"
    df2022["Code departement"] = pd.to_numeric(df2022["Code departement"], errors="coerce")
    df2018["Code departement"] = pd.to_numeric(df2018["Code departement"], errors="coerce")
    # Calcul du prix moyen du mètre carré par département pour 2022 et 2018
    prix_moyen_m2_2022 = df2022.groupby("Code departement")["Valeur fonciere"].sum() / df2022.groupby("Code departement")["Surface terrain"].sum()
    prix_moyen_m2_2018 = df2018.groupby("Code departement")["Valeur fonciere"].sum() / df2018.groupby("Code departement")["Surface terrain"].sum()
    # Tracé des graphiques côte à côte
    fig, axs = plt.subplots(ncols=2, figsize=(12, 6))
    axs[0].bar(prix_moyen_m2_2018.index, prix_moyen_m2_2018.values)
    axs[0].set_title("Prix moyen du mètre carré en 2018")
    axs[0].set_xlabel("Département")
    axs[0].set_ylabel("Prix moyen du mètre carré")
    axs[0].set_xlim(0, 96)
    axs[0].set_ylim(0, 3000)
    axs[1].bar(prix_moyen_m2_2022.index, prix_moyen_m2_2022.values)
    axs[1].set_title("Prix moyen du mètre carré en 2022")
    axs[1].set_xlabel("Département")
    axs[1].set_ylabel("Prix moyen du mètre carré")
    axs[1].set_xlim(0, 96)
    axs[1].set_ylim(0, 3000)
    # Affichage des graphiques
    plt.tight_layout()
    plot_html = mpld3.fig_to_html(plt.gcf())
    
    return plot_html


def visu28():
    df2018 = data2018mini
    df2022 = data2022mini
    # Convertir la colonne "Valeur fonciere" en type numérique
    df2018['Valeur fonciere'] = pd.to_numeric(df2018['Valeur fonciere'], errors='coerce')
    df2022['Valeur fonciere'] = pd.to_numeric(df2022['Valeur fonciere'], errors='coerce')
    # Grouper les données par mois et calculer le prix moyen pour 2018 et 2022
    prix_moyen_2018 = df2018.groupby('Date mutation')['Valeur fonciere'].mean()
    prix_moyen_2022 = df2022.groupby('Date mutation')['Valeur fonciere'].mean()
    # Créer une figure et deux axes pour les graphiques
    fig, ax1 = plt.subplots(figsize=(15, 6))
    ax2 = ax1.twiny()
    # Tracer la courbe pour 2018 sur le premier axe
    ax1.plot(prix_moyen_2018.index, prix_moyen_2018.values, label='2018', color='blue')
    # Tracer la courbe pour 2022 sur le deuxième axe
    ax2.plot(prix_moyen_2022.index, prix_moyen_2022.values, label='2022', color='red')
    # Configurer les axes et les étiquettes
    ax1.set_xlabel('Mois (2018)', color='blue')
    ax1.tick_params(axis='x', colors='blue')
    ax1.set_ylabel('Prix moyen des transactions immobilières', color='blue')
    ax2.set_xlabel('Mois (2022)', color='red')
    ax2.tick_params(axis='x', colors='red')
    # Ajouter une légende
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper left')
    # Ajouter le titre du graphique
    plt.title('Évolution du prix moyen des transactions immobilières (2018 vs 2022)')
    plot_html = mpld3.fig_to_html(plt.gcf())
    
    return plot_html


def visu29():
    # Set the figure size
    plt.figure(figsize=(10, 6))
    # Create the histogram for 2018 data
    plt.hist(data2018mini["Valeur fonciere"], bins=500, alpha=0.5, color='blue', label='2018')
    # Create the histogram for 2022 data
    plt.hist(data2022mini["Valeur fonciere"], bins=500, alpha=0.5, color='red', label='2022')
    # Add title and labels
    plt.ylim(0, 3000000)
    plt.xlim(0, 3000000)
    plt.title("Distribution de la valeur foncière par an")
    plt.xlabel("Valeur foncière (en millions d'euros)")
    plt.ylabel("Nombre de Transactions")
    # Add legend
    plt.legend()
    plot_html = mpld3.fig_to_html(plt.gcf())
    
    return plot_html


def visu30():
    """
    # Read in the GeoJSON file for the department borders
    france_map = gpd.read_file("https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-version-simplifiee.geojson")
    
    # Calculate the count of 'Maison' by department for 2022 and sort in descending order
    maison_counts_2022 = data2022mini[data2022mini['Type local'] == 'Maison']['Code departement'].astype(str).value_counts().sort_index()
    
    # Calculate the count of 'Maison' by department for 2018 and sort in descending order
    maison_counts_2018 = data2018mini[data2018mini['Type local'] == 'Maison']['Code departement'].astype(str).value_counts().sort_index()
    
    # Calculate the 90th percentile of the "Maison" values for both years
    q90_2022 = maison_counts_2022.quantile(0.8)
    q90_2018 = maison_counts_2018.quantile(0.8)
    
    # Assign a color gradient to the departments based on the count of 'Maison' for 2022
    france_map = france_map.rename(columns={'code': 'Code departement'})
    france_map['color_2022'] = france_map['Code departement'].apply(lambda x: cm.get_cmap('magma')(maison_counts_2022.get(x, 0)/q90_2022))
    
    # Assign a color gradient to the departments based on the count of 'Maison' for 2018
    france_map['color_2018'] = france_map['Code departement'].apply(lambda x: cm.get_cmap('magma')(maison_counts_2018.get(x, 0)/q90_2018))
    
    # Create the subplots for the two maps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot the map for 2018 with the colored departments
    france_map.plot(ax=ax1, color=france_map['color_2018'], edgecolor='white', linewidth=0.5)
    ax1.set_title("Nombre de transaction avec des maisons par département en 2018")
    ax1.axis('off')
    
    # Plot the map for 2022 with the colored departments
    france_map.plot(ax=ax2, color=france_map['color_2022'], edgecolor='white', linewidth=0.5)
    ax2.set_title("Nombre de transaction avec des maisons par département en 2022")
    ax2.axis('off')
    
    
    # Add colorbar
    vminT = min(maison_counts_2018.min(), maison_counts_2022.min())
    vmaxT = max(maison_counts_2018.max(), maison_counts_2022.max())
    
    sm_2018 = plt.cm.ScalarMappable(cmap=cm.get_cmap('magma'), norm=plt.Normalize(vmin=vminT, vmax=vmaxT))
    sm_2018._A = []
    cbar_2018 = plt.colorbar(sm_2018, ax=ax2, fraction=0.03, pad=0.04)
    cbar_2018.ax.set_ylabel('Nombre de maisons', rotation=270, labelpad=20)
    sm_2022 = plt.cm.ScalarMappable(cmap=cm.get_cmap('magma'), norm=plt.Normalize(vmin=vminT, vmax=vmaxT))
    sm_2022._A = []
    cbar_2022 = plt.colorbar(sm_2022, ax=ax1, fraction=0.03, pad=0.04)
    cbar_2022.ax.set_ylabel('Nombre de maisons', rotation=270, labelpad=20)
    
    plt.subplots_adjust(wspace=0.05)
    plt.show()
    plt.savefig('testGraph1/static/figure29.png')  # Save the figure to a file

    """
    #plot_html = mpld3.fig_to_html(plt.gcf())
    
    return 0






