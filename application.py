from flask import Flask, render_template
from apscheduler.schedulers.background import BackgroundScheduler
import re
import pandas as pd
import os
from datetime import datetime, timedelta
import requests_cache
from retrying import retry
import openmeteo_requests
import requests, shelve, json, time, unicodedata, os, sys, errno
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import locale

from sqlalchemy import create_engine, inspect, Column, String, Integer, MetaData, Table, DateTime
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Float, ForeignKey, Table
from sqlalchemy.orm import sessionmaker, relationship, backref
from sqlalchemy import exc
from sqlalchemy.sql import func
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy import UniqueConstraint


app = Flask(__name__)

scheduler = BackgroundScheduler()

engine = create_engine('sqlite:///data/Thermometer.db')

# Définir une base pour les modèles
Base = declarative_base()

# Définition des modèles Sqlachemy
class XiaomiTemperature(Base):
    __tablename__ = 'xiaomi_temperature'
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False)
    mac_address = Column(String, nullable=False)
    room = Column(String, nullable=False)
    temperature = Column(Float, nullable=False)
    humidity = Column(Float, nullable=False)

# Définition des modèles Sqlachemy pour Méteo API
class MeteoAPI30min(Base):
    __tablename__ = 'meteo_api_30min'
    date = Column(DateTime, primary_key=True, nullable=False)  # Utiliser `date` comme clé primaire
    temperature_2m = Column(Float, nullable=False)
    sunshine_duration = Column(Float, nullable=False)
    shortwave_radiation = Column(Float, nullable=False)
    direct_radiation = Column(Float, nullable=False)

class XiaomiTemperature30minModele(Base):
    __tablename__ = 'xiaomi_temperature_30_min'
    __table_args__ = (UniqueConstraint('timestamp', 'room', name='unique_timestamp_room'),)
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False)
    room = Column(String, nullable=False)  # Correspond à "Pièce"
    mac_address = Column(String, nullable=False)  # Correspond à "Adresse MAC"
    avg_temperature = Column(Float, nullable=True)  # Correspond à "moyenne_temperature"
    std_temperature = Column(Float, nullable=True)  # Correspond à "ecart_type_temperature"
    avg_humidity = Column(Float, nullable=True)  # Correspond à "moyenne_humid"
    std_humidity = Column(Float, nullable=True)  # Correspond à "ecart_type_humid"

class CozyTouchTemperatureModele(Base):
    __tablename__ = 'cozy_touch_table'

    id = Column(Integer, primary_key=True, autoincrement=True)  # Clé primaire
    timestamp = Column(DateTime, nullable=False)  # Colonne pour la date et l'heure
    piece = Column(String, nullable=False)  # Nom de la pièce
    temperature = Column(Float, nullable=False)  # Température actuelle
    target_temperature = Column(Float, nullable=False)  # Température cible
    consumption = Column(Integer, nullable=False)  # Consommation en unités

    # Définir la contrainte d'unicité sur `timestamp` et `piece`
    __table_args__ = (
        UniqueConstraint('timestamp', 'piece', name='uix_timestamp_piece'),
    )


# Configurer une session pour interagir avec la base
Session = sessionmaker(bind=engine)
session = Session()

# Assurez-vous que la localisation en français est activée pour afficher les jours et mois en français
locale.setlocale(locale.LC_TIME, 'fr_FR.UTF-8')

def CleaningXiaomiTemp(deltajour):
    
    # Spécifiez ici l'adresse MAC que vous souhaitez suivre
    target_mac_addresses = [
    ("A4:C1:38:C9:36:78","Chambre 2"),
    ("A4:C1:38:39:A8:57","Salon")  # Ajoutez d'autres adresses ici
    # "AUTRE_ADRESSE_MAC"
    ]

    data_folder = '/home/pi/CozyTouchAPI//MiTemperature2/data/'

    # Lister tous les fichiers dans le dossier
    files = os.listdir(data_folder)

    # Liste pour stocker les fichiers et leurs timestamps
    file_timestamps = []


    for file in files:

        # Extraire la partie date_heure du nom de fichier
        # On supprime le préfixe 'data_' et l'extension '.log'
        date_time_str = file[len('data_'):-len('.log')]

        # Définir le format de la date et de l'heure
        date_format = '%Y-%m-%d_%H-%M-%S'

        # Convertir la chaîne en objet datetime
        timestamp = datetime.strptime(date_time_str, date_format)

        file_timestamps.append((file, timestamp))

    # Liste pour stocker les résultats
    data = []

    # Lecture des données dans les fichiers
    for filename, time in file_timestamps:
        full_path = os.path.join(data_folder, filename)

        # Ouvrir le fichier en mode lecture
        with open(full_path, 'r') as file:
            lines = file.readlines()  # Lire toutes les lignes du fichier

            # Parcourir les lignes avec leur index
            for i in range(len(lines)):
                # Chercher l'adresse MAC dans la ligne actuelle
                for target_mac_address, room_name in target_mac_addresses:
                    if f"{target_mac_address}" in lines[i]:
                        # La ligne suivante contiendra la température et l'humidité
                        if i + 1 < len(lines):  # Vérifiez que la ligne suivante existe
                            temp_line = lines[i + 1]
                            
                            # Chercher la température et l'humidité dans la ligne suivante
                            match = re.search(r"Temperature:\s*([\d.]+)", temp_line)
                            
                            humidity_line = lines[i + 2]
                            match2 = re.search(r"Humidity:\s*(\d+)", humidity_line)

                            # Si une correspondance est trouvée, extraire les données
                            if match:
                                temperature = match.group(1)
                                humidity = match2.group(1)
                                
                                # Ajouter les résultats à la liste
                                data.append({
                                    'Timestamp' : time,
                                    'Adresse MAC': target_mac_address,
                                    'Pièce': room_name,
                                    'Température': temperature,
                                    'Humidité': humidity
                                })

    # Créer un DataFrame à partir des données
    df = pd.DataFrame(data)
    df = df.drop_duplicates(subset=['Adresse MAC', 'Timestamp'])
    df = df.sort_values(by='Timestamp')
    df['Température'] = pd.to_numeric(df['Température'], errors='coerce')
    df['Humidité'] = pd.to_numeric(df['Humidité'], errors='coerce')

    # Insertion dans la base de donnée sqlite sans les doublons
    for _, row in df.iterrows():
        exists = session.query(XiaomiTemperature).filter_by(
            timestamp=row['Timestamp'],
            mac_address=row['Adresse MAC'],
            room=row['Pièce'],
            temperature=row['Température'],
            humidity=row['Humidité']
        ).first()

        if not exists:
            new_entry = XiaomiTemperature(
                timestamp=row['Timestamp'],
                mac_address=row['Adresse MAC'],
                room=row['Pièce'],
                temperature=row['Température'],
                humidity=row['Humidité']
            )
            session.add(new_entry)
            session.commit()  # Commit pour chaque nouvelle entrée


    # Append des données à l'existant
    # df_hist = pd.read_csv('/home/pi/CozyTouchAPI/data/XiaomiTemperature_data.csv')
    # df_hist['Timestamp'] = pd.to_datetime(df_hist['Timestamp'])

    # df_new = pd.concat([df, df_hist], ignore_index=True)

    # df_new = df_new.drop_duplicates(subset=['Adresse MAC', 'Timestamp'])
    # df_new = df_new.sort_values(by='Timestamp')

    # Exporter le DataFrame en fichier CSV
    # df_new.to_csv('/home/pi/CozyTouchAPI/data/XiaomiTemperature_data.csv', index=False)

    # Supprimer les fichiers de plus de un jour

    # df_cleaning = pd.read_csv('/home/pi/CozyTouchAPI/data/XiaomiTemperature_data.csv')
    # df_cleaning['Timestamp'] = pd.to_datetime(df_cleaning['Timestamp'])

    # Date et heure actuelle
    current_date = datetime.now()

    # Date et heure d'il y a un jour
    yesterday_date = current_date - timedelta(days=deltajour)

    for file, timestamp in file_timestamps:
        # Vérifie si le timestamp est plus ancien que la date d'hier
        if timestamp < yesterday_date:
            full_path = os.path.join(data_folder, file)
            if os.path.exists(full_path):
                print(full_path)  # Affiche le chemin complet du fichier à supprimer
                os.remove(full_path)  # Supprime le fichier


def MeteoAPI30minfunc():
    # Configurer la session de cache
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)

    # Configurer le retry avec urllib3 pour le réessai automatique
    retry_strategy = Retry(
        total=5,
        backoff_factor=0.2,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    cache_session.mount("https://", adapter)
    cache_session.mount("http://", adapter)

    openmeteo = openmeteo_requests.Client(session=cache_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": 48.8574,
        "longitude": 2.3795,
        "minutely_15": ["temperature_2m", "sunshine_duration", "shortwave_radiation", "direct_radiation"],
        "timezone": "Europe/Berlin",
        "past_days": 1,
        "past_minutely_15": 96,
        "forecast_days": 1,
        "forecast_minutely_15": 96
    }

    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    # print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    # print(f"Elevation {response.Elevation()} m asl")
    # print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    # print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process minutely_15 data. The order of variables needs to be the same as requested.
    minutely_15 = response.Minutely15()
    minutely_15_temperature_2m = minutely_15.Variables(0).ValuesAsNumpy()
    minutely_15_sunshine_duration = minutely_15.Variables(1).ValuesAsNumpy()
    minutely_15_shortwave_radiation = minutely_15.Variables(2).ValuesAsNumpy()
    minutely_15_direct_radiation = minutely_15.Variables(3).ValuesAsNumpy()

    minutely_15_data = {"date": pd.date_range(
        start = pd.to_datetime(minutely_15.Time(), unit = "s", utc = True).tz_convert("Europe/Paris"),
        end = pd.to_datetime(minutely_15.TimeEnd(), unit = "s", utc = True).tz_convert("Europe/Paris"),
        freq = pd.Timedelta(seconds = minutely_15.Interval()),
        inclusive = "left"
    )}
    minutely_15_data["temperature_2m"] = minutely_15_temperature_2m
    minutely_15_data["sunshine_duration"] = minutely_15_sunshine_duration
    minutely_15_data["shortwave_radiation"] = minutely_15_shortwave_radiation
    minutely_15_data["direct_radiation"] = minutely_15_direct_radiation

    minutely_15_dataframe = pd.DataFrame(data = minutely_15_data)
 
    minutely_15_dataframe.set_index('date', inplace=True)

    # Faire un resample toutes les 30 minutes avec une agrégation personnalisée
    df_30min = minutely_15_dataframe.resample('30T').agg({
        'temperature_2m': 'mean',           # Moyenne des températures
        'sunshine_duration': 'sum',         # Somme de la durée d'ensoleillement
        'shortwave_radiation': 'sum',       # Somme de la radiation à ondes courtes
        'direct_radiation': 'sum'           # Somme de la radiation directe
    })
    
    df_30min['temperature_2m'] = df_30min['temperature_2m'].round(2)

    # df_hist_30min = pd.read_csv('/home/pi/CozyTouchAPI/data/MeteoAPI30min.csv')

    # df_hist_30min['date'] = pd.to_datetime(df_hist_30min['date'])
    # df_hist_30min.set_index('date', inplace=True)

    # df_new_30min = pd.concat([df_30min, df_hist_30min], ignore_index=False)

    # df_new_30min = df_new_30min.reset_index().drop_duplicates(subset=['date'])

    # df_new_30min = df_new_30min.sort_values(by='date')

    # Exporter le DataFrame en fichier CSV
    # df_new_30min.to_csv('/home/pi/CozyTouchAPI/data/MeteoAPI30min.csv', index=False)

    df_30min = df_30min.reset_index()

    # Insérer ou mettre à jour chaque ligne
    for _, row in df_30min.iterrows():
        stmt = insert(MeteoAPI30min).values(
            date=row["date"],
            temperature_2m=row["temperature_2m"],
            sunshine_duration=row["sunshine_duration"],
            shortwave_radiation=row["shortwave_radiation"],
            direct_radiation=row["direct_radiation"]
        )
        # On conflict, do update
        stmt = stmt.on_conflict_do_update(
            index_elements=["date"],  # Colonne(s) utilisée(s) pour détecter les conflits
            set_={
                "temperature_2m": stmt.excluded.temperature_2m,
                "sunshine_duration": stmt.excluded.sunshine_duration,
                "shortwave_radiation": stmt.excluded.shortwave_radiation,
                "direct_radiation": stmt.excluded.direct_radiation,
            }
        )
        session.execute(stmt)

    session.commit()

def XiaomiTemperature30min():
    # df = pd.read_csv('/home/pi/CozyTouchAPI/data/XiaomiTemperature_data.csv')
    # Charger toutes les données de la table XiaomiTemperature

    last_6_hours = datetime.now() - timedelta(hours=6)

    results = session.query(XiaomiTemperature).filter(XiaomiTemperature.timestamp >= last_6_hours).all()

    data = [
        {
            "Timestamp": row.timestamp,
            "Adresse MAC": row.mac_address,
            "Pièce": row.room,
            "Température": row.temperature,
            "Humidité": row.humidity
        }
        for row in results
    ]

    df = pd.DataFrame(data)

    # Convertir la colonne 'Timestamp' en type datetime avant de définir comme index
    # df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Vérifier et convertir les types de colonnes si nécessaire
    # df['Température'] = pd.to_numeric(df['Température'], errors='coerce')
    # df['Humidité'] = pd.to_numeric(df['Humidité'], errors='coerce')

    # Vérification des types de données après conversion
    # print(df.dtypes)

    # Définir 'Timestamp' comme index
    df.set_index('Timestamp', inplace=True)

    # Agréger les données toutes les 30 minutes en prenant la moyenne et l'écart type, en groupant par 'Pièce' et 'Adresse MAC'
    # df_aggregated = df.groupby(['Pièce', 'Adresse MAC']).resample('30min')['Température', 'Humidité'].agg(['mean']).reset_index()

    df_aggregated = df.groupby(['Pièce', 'Adresse MAC']).resample('30min').agg(
        moyenne_temperature=('Température', 'mean'),
        ecart_type_temperature=('Température', 'std'),
        moyenne_humid=('Humidité', 'mean'),
        ecart_type_humid=('Humidité', 'std')
    ).reset_index()
    # Afficher le DataFrame agrégé

    df_aggregated['moyenne_temperature'] = df_aggregated['moyenne_temperature'].round(2)
    df_aggregated['moyenne_humid'] = df_aggregated['moyenne_humid'].round(2)

    # Parcourir les lignes du DataFrame
    for _, row in df_aggregated.iterrows():
        stmt = insert(XiaomiTemperature30minModele).values(
            timestamp=row["Timestamp"],
            room=row["Pièce"],
            mac_address=row["Adresse MAC"],
            avg_temperature=row["moyenne_temperature"],
            std_temperature=row["ecart_type_temperature"],
            avg_humidity=row["moyenne_humid"],
            std_humidity=row["ecart_type_humid"]
        )
        # En cas de conflit sur (timestamp, room), mettre à jour les autres colonnes
        stmt = stmt.on_conflict_do_update(
            index_elements=["timestamp", "room"],  # Clé unique utilisée pour détecter les conflits
            set_={
                "mac_address": stmt.excluded.mac_address,
                "avg_temperature": stmt.excluded.avg_temperature,
                "std_temperature": stmt.excluded.std_temperature,
                "avg_humidity": stmt.excluded.avg_humidity,
                "std_humidity": stmt.excluded.std_humidity,
            }
        )
        # Exécuter la requête
        session.execute(stmt)

    # Commit pour sauvegarder les modifications
    session.commit()

    # Calculer la limite de 7 jours
    seven_days_ago = datetime.now() - timedelta(days=7)

    # Supprimer les lignes avec un timestamp plus ancien que 7 jours
    try:
        # Filtrer et supprimer
        session.query(XiaomiTemperature).filter(XiaomiTemperature.timestamp < seven_days_ago).delete(synchronize_session=False)
        
        # Commit pour appliquer les changements
        session.commit()
        print("Les lignes de plus de 7 jours ont été supprimées avec succès.")
    except Exception as e:
        # En cas d'erreur
        session.rollback()
        print(f"Erreur lors de la suppression : {e}")

    # df_hist_30min = pd.read_csv('/home/pi/CozyTouchAPI/data/XiaomiTemperature30min_data.csv')

    # df_hist_30min['Timestamp'] = pd.to_datetime(df_hist_30min['Timestamp'])

    # df_new_30min = pd.concat([df_aggregated, df_hist_30min], ignore_index=False)

    # df_new_30min = df_new_30min.drop_duplicates(subset=['Timestamp','Pièce'])

    # df_new_30min = df_new_30min.sort_values(by='Timestamp')

    # Exporter le DataFrame en fichier CSV
    # df_new_30min.to_csv('/home/pi/CozyTouchAPI/data/XiaomiTemperature30min_data.csv', index=False)



    # Suppression des données détaillées de plus d'une semaine

    # Calculer la date une semaine avant la date actuelle
    # one_week_ago = datetime.now() - timedelta(weeks=1)

    # Filtrer pour ne garder que les lignes dont l'index est postérieur à one_week_ago
    # df_filtered = df[df.index > one_week_ago]

    # df_filtered.to_csv('/home/pi/CozyTouchAPI/data/XiaomiTemperature_data.csv', index=True)


def CozyTouch30min():
    df = pd.read_csv('/home/pi/CozyTouchAPI/data/CozyTouch_data.csv')

    # Convertir la colonne 'Timestamp' en type datetime avant de définir comme index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Définir 'Timestamp' comme index
    df.set_index('timestamp', inplace=True)

    df_aggregated = df.groupby(['piece']).resample('30min').agg(
        moyenne_temperature=('temperature', 'mean'),
        ecart_type_temperature=('temperature', 'std'),
        temperature_cible=('temperature cible', 'mean'),
        consommation=('consommation', 'max')
    ).reset_index()

    df_aggregated['moyenne_temperature'] = df_aggregated['moyenne_temperature'].round(2)

    df_hist_30min = pd.read_csv('/home/pi/CozyTouchAPI/data/CozyTouch30min_data.csv')

    df_hist_30min['timestamp'] = pd.to_datetime(df_hist_30min['timestamp'])


    df_new_30min = pd.concat([df_aggregated, df_hist_30min], ignore_index=False)

    df_new_30min = df_new_30min.drop_duplicates(subset=['timestamp','piece'])

    df_new_30min = df_new_30min.sort_values(by='timestamp')

    # Exporter le DataFrame en fichier CSV
    df_new_30min.to_csv('/home/pi/CozyTouchAPI/data/CozyTouch30min_data.csv', index=False)

    # Suppression des données détaillées de plus d'une semaine

    # Calculer la date une semaine avant la date actuelle
    one_week_ago = datetime.now() - timedelta(weeks=1)

    # Filtrer pour ne garder que les lignes dont l'index est postérieur à one_week_ago
    df_filtered = df[df.index > one_week_ago]

    df_filtered.to_csv('/home/pi/CozyTouchAPI/data/CozyTouch_data.csv', index=True)

# Dictionnaire pour stocker les données de température
temperature_data = {
    "Salon": {"temperature": None, "humidite": None,"timestamp": None, "temperature cible": None},
    "Chambre 2": {"temperature": None, "humidite": None, "timestamp": None, "temperature cible": None},
    "Exterieur": {"temperature": None}
    }


def format_relative_time(timestamp):
    """Formatte une durée relative par rapport au timestamp."""
    now = datetime.now()
    diff = now - timestamp
    seconds = diff.total_seconds()

    if seconds < 60:
        return "il y a moins d'une minute"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        return f"il y a {minutes} minute{'s' if minutes > 1 else ''}"
    elif seconds < 86400:
        hours = int(seconds // 3600)
        return f"il y a {hours} heure{'s' if hours > 1 else ''}"
    else:
        return timestamp.strftime('%d/%m/%Y %H:%M')


def update_temperature_data():
    """Fonction qui met à jour les données de température pour chaque pièce."""
    global temperature_data

    current_date = datetime.now()

    # Récupérer data Xiaomi
    # df = pd.read_csv('/home/pi/CozyTouchAPI/data/XiaomiTemperature_data.csv')
    # df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    # df = df.sort_values(by='Timestamp')

    # Récupérer l'écriture la plus récente pour "Salon"
    latest_record_salon = (
        session.query(XiaomiTemperature)
        .filter(XiaomiTemperature.room == "Salon")
        .order_by(XiaomiTemperature.timestamp.desc())  # Trier par timestamp décroissant
        .first()  # Récupérer la première ligne
    )
   
    latest_record_chambre2 = (
        session.query(XiaomiTemperature)
        .filter(XiaomiTemperature.room == "Chambre 2")
        .order_by(XiaomiTemperature.timestamp.desc())  # Trier par timestamp décroissant
        .first()  # Récupérer la première ligne
    )



    # Récupérer la dernière température pour chaque pièce pour le Xiaomi
    temperature_data["Salon"]["temperature"] = latest_record_salon.temperature
    temperature_data["Salon"]["humidite"] = latest_record_salon.humidity
    temperature_data["Salon"]["timestamp"] = latest_record_salon.timestamp
    temperature_data["Chambre 2"]["temperature"] = latest_record_chambre2.temperature
    temperature_data["Chambre 2"]["humidite"] = latest_record_chambre2.humidity
    temperature_data["Chambre 2"]["timestamp"] = latest_record_chambre2.timestamp



    # Récupérer data CozyTouch
    # df = pd.read_csv('/home/pi/CozyTouchAPI/data/CozyTouch_data.csv')
    # df['timestamp'] = pd.to_datetime(df['timestamp'])
    # df = df.sort_values(by='timestamp')
        
    latest_record_cozy_chambre2 = (
        session.query(CozyTouchTemperatureModele)
        .filter(CozyTouchTemperatureModele.piece== "Chambre 2")
        .order_by(CozyTouchTemperatureModele.timestamp.desc())  # Trier par timestamp décroissant
        .first()  # Récupérer la première ligne
    )

    latest_record_cozy_salon = (
        session.query(CozyTouchTemperatureModele)
        .filter(CozyTouchTemperatureModele.piece== "Chambre 2")
        .order_by(CozyTouchTemperatureModele.timestamp.desc())  # Trier par timestamp décroissant
        .first()  # Récupérer la première ligne
    )

    temperature_data["Salon"]["temperature cible"] = latest_record_cozy_salon.target_temperature
    temperature_data["Chambre 2"]["temperature cible"] = latest_record_cozy_chambre2.target_temperature

    # Récupérer température extérieur
    # Requête pour trouver la ligne avec la date la plus proche
    
    closest_entry = (
        session.query(MeteoAPI30min)
        .order_by(func.abs(func.julianday(MeteoAPI30min.date) - func.julianday(current_date)))
        .first()
    )
    temperature_data["Exterieur"]["temperature"] = round(closest_entry.temperature_2m,1)




@app.route('/')
def hello():
    # CleaningXiaomiTemp(1)
    update_temperature_data()

    """Vue principale qui affiche les données de température avec un formatage relatif."""
    formatted_data = {
        "Salon": {
            "temperature": temperature_data["Salon"]["temperature"],
            "humidité": temperature_data["Salon"]["humidite"],
            "temperature cible": temperature_data["Salon"]["temperature cible"],
            "relative_time": format_relative_time(temperature_data["Salon"]["timestamp"]) if temperature_data["Salon"]["timestamp"] else None
        },
        "Chambre 2": {
            "temperature": temperature_data["Chambre 2"]["temperature"],
            "humidité": temperature_data["Chambre 2"]["humidite"],
            "temperature cible": temperature_data["Chambre 2"]["temperature cible"],
            "relative_time": format_relative_time(temperature_data["Chambre 2"]["timestamp"]) if temperature_data["Chambre 2"]["timestamp"] else None
        },
        "Exterieur" : {
            "temperature" : temperature_data["Exterieur"]["temperature"]
        }
    }

    # Formater la date actuelle pour l'afficher sous le titre
    date_aujourdhui = datetime.now().strftime('%A %d %B %Y %Hh%M')

    return render_template('index.html', temperature_data=formatted_data, date_aujourdhui=date_aujourdhui)


if __name__ == '__main__':

    # Planifie la tâche pour qu'elle s'exécute avec une fréquence
    scheduler.add_job(func=CleaningXiaomiTemp, trigger='interval', minutes=5, args=[1])
    scheduler.add_job(func=MeteoAPI30minfunc, trigger='interval', minutes=30)
    scheduler.add_job(func=XiaomiTemperature30min, trigger='interval', minutes=30)
    scheduler.add_job(func=CozyTouch30min, trigger='interval', minutes=30)
    scheduler.start()

    app.run(host='0.0.0.0', port=5075, debug=True)