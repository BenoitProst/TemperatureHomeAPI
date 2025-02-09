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

from pathlib import Path

from sqlalchemy import create_engine, inspect, Column, String, Integer, MetaData, Table, DateTime
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Float, ForeignKey, Table
from sqlalchemy.orm import sessionmaker, relationship, backref, scoped_session
from sqlalchemy import exc
from sqlalchemy.sql import func
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy import UniqueConstraint
from sqlalchemy.exc import OperationalError


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


class CozyTouchTemperature30minModele(Base):
    __tablename__ = 'cozy_touch_temperature_30min'  # Nom de la table dans la base de données

    id = Column(Integer, primary_key=True, autoincrement=True)  # Identifiant unique
    piece = Column(String, nullable=False)  # Nom de la pièce
    timestamp = Column(DateTime, nullable=False)  # Horodatage
    moyenne_temperature = Column(Float, nullable=True)  # Température moyenne
    ecart_type_temperature = Column(Float, nullable=True)  # Écart-type de la température
    temperature_cible = Column(Float, nullable=True)  # Température cible
    consommation = Column(Float, nullable=True)  # Consommation en énergie

    # Contrainte unique sur le couple (timestamp, piece)
    __table_args__ = (UniqueConstraint('timestamp', 'piece', name='uix_timestamp_piece'),)


# Configurer une session pour interagir avec la base
Session = sessionmaker(bind=engine)
session = Session()

# Création d'une session scoped
SessionSc = scoped_session(sessionmaker(bind=engine))

# Assurez-vous que la localisation en français est activée pour afficher les jours et mois en français
locale.setlocale(locale.LC_TIME, 'fr_FR.UTF-8')


def CleaningXiaomiTemp(deltajour):
    session = SessionSc()  # Récupération de la session isolée

    try:
        # Définition des adresses MAC cibles
        target_mac_addresses = {
            "A4:C1:38:C9:36:78": "Chambre 2",
            "A4:C1:38:39:A8:57": "Salon"
        }

        data_folder = Path('/home/pi/CozyTouchAPI/MiTemperature2/data/')
        file_timestamps = []

        # Lister et parser les fichiers du dossier
        for file in data_folder.glob("data_*.log"):
            try:
                date_time_str = file.stem[len("data_"):]  # Extraction de la date
                timestamp = datetime.strptime(date_time_str, "%Y-%m-%d_%H-%M-%S")
                file_timestamps.append((file, timestamp))
            except ValueError:
                print(f"Ignoré : {file}")

        data = []

        # Lecture des fichiers et extraction des données
        for filepath, timestamp in file_timestamps:
            with open(filepath, 'r') as file:
                lines = file.readlines()

            for i, line in enumerate(lines):
                for mac, room in target_mac_addresses.items():
                    if mac in line and i + 2 < len(lines):
                        temp_match = re.search(r"Temperature:\s*([\d.]+)", lines[i + 1])
                        hum_match = re.search(r"Humidity:\s*(\d+)", lines[i + 2])

                        if temp_match and hum_match:
                            data.append({
                                "Timestamp": timestamp,
                                "Adresse MAC": mac,
                                "Pièce": room,
                                "Température": float(temp_match.group(1)),
                                "Humidité": int(hum_match.group(1))
                            })

        # Création du DataFrame
        if not data:
            return

        df = pd.DataFrame(data).drop_duplicates(subset=["Adresse MAC", "Timestamp"])
        df.sort_values("Timestamp", inplace=True)

        df.rename(columns={
            "Timestamp": "timestamp",
            "Adresse MAC": "mac_address",
            "Pièce": "room",
            "Température": "temperature",
            "Humidité": "humidity"
        }, inplace=True)
        
        # Insertion optimisée en base de données
        insert_data = df.to_dict(orient="records")
        if insert_data:
            stmt = insert(XiaomiTemperature).values(insert_data)
            stmt = stmt.on_conflict_do_nothing()  # Ignore les doublons existants
            session.execute(stmt)

        session.commit()

        # Suppression des fichiers de plus de `deltajour` jours
        date_limite = datetime.now() - timedelta(days=deltajour)
        for filepath, timestamp in file_timestamps:
            if timestamp < date_limite:
                filepath.unlink(missing_ok=True)  # Supprime le fichier

    except Exception as e:
        session.rollback()
        print(f"Erreur dans CleaningXiaomiTemp : {e}")

    finally:
        session.close()   # Nettoyage de la session après usage

def MeteoAPI30minfunc():
    """Récupère les données météo et les stocke en base en évitant les conflits."""
    
    # Configuration de la session cache
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)

    retry_strategy = Retry(
        total=5,
        backoff_factor=0.2,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    cache_session.mount("https://", adapter)
    cache_session.mount("http://", adapter)

    openmeteo = openmeteo_requests.Client(session=cache_session)
    
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

    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        
        minutely_15 = response.Minutely15()
        minutely_15_data = {
            "date": pd.date_range(
                start=pd.to_datetime(minutely_15.Time(), unit="s", utc=True).tz_convert("Europe/Paris"),
                end=pd.to_datetime(minutely_15.TimeEnd(), unit="s", utc=True).tz_convert("Europe/Paris"),
                freq=pd.Timedelta(seconds=minutely_15.Interval()),
                inclusive="left"
            ),
            "temperature_2m": minutely_15.Variables(0).ValuesAsNumpy(),
            "sunshine_duration": minutely_15.Variables(1).ValuesAsNumpy(),
            "shortwave_radiation": minutely_15.Variables(2).ValuesAsNumpy(),
            "direct_radiation": minutely_15.Variables(3).ValuesAsNumpy(),
        }

        df_30min = pd.DataFrame(minutely_15_data).set_index('date').resample('30T').agg({
            'temperature_2m': 'mean',
            'sunshine_duration': 'sum',
            'shortwave_radiation': 'sum',
            'direct_radiation': 'sum'
        })

        df_30min['temperature_2m'] = df_30min['temperature_2m'].round(2)
        df_30min = df_30min.reset_index()

        session = SessionSc()

        # Création de la liste des dictionnaires pour un insert en bulk
        data_to_insert = df_30min.to_dict(orient="records")

        if data_to_insert:
            stmt = insert(MeteoAPI30min).values(data_to_insert)
            stmt = stmt.on_conflict_do_update(
                index_elements=["date"],
                set_={
                    "temperature_2m": stmt.excluded.temperature_2m,
                    "sunshine_duration": stmt.excluded.sunshine_duration,
                    "shortwave_radiation": stmt.excluded.shortwave_radiation,
                    "direct_radiation": stmt.excluded.direct_radiation,
                }
            )
            session.execute(stmt)
            session.commit()

    except OperationalError as e:
        print(f"Erreur SQLAlchemy: {e}")
        session.rollback()
    finally:
        session.close()

def XiaomiTemperature30min():
    session = SessionSc()  # Récupération de la session isolée

    try:
        # Récupérer les données des 6 dernières heures
        last_6_hours = datetime.now() - timedelta(hours=6)
        results = session.query(XiaomiTemperature).filter(XiaomiTemperature.timestamp >= last_6_hours).all()

        # Transformer les résultats en DataFrame
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

        if df.empty:  # Éviter les erreurs si aucune donnée n'est trouvée
            return  

        df.set_index('Timestamp', inplace=True)

        # Agréger les données toutes les 30 minutes
        df_aggregated = df.groupby(['Pièce', 'Adresse MAC']).resample('30min').agg(
            moyenne_temperature=('Température', 'mean'),
            ecart_type_temperature=('Température', 'std'),
            moyenne_humid=('Humidité', 'mean'),
            ecart_type_humid=('Humidité', 'std')
        ).reset_index()

        df_aggregated['moyenne_temperature'] = df_aggregated['moyenne_temperature'].round(2)
        df_aggregated['moyenne_humid'] = df_aggregated['moyenne_humid'].round(2)
        
        df_aggregated.rename(columns={
            "Pièce": "room",
            "Adresse MAC": "mac_address",
            "moyenne_temperature": "avg_temperature",
            "ecart_type_temperature": "std_temperature",
            "moyenne_humid": "avg_humidity",
            "ecart_type_humid": "std_humidity",
            "Timestamp": "timestamp"
        }, inplace=True)
        
        # Préparer les données pour un bulk insert avec upsert (on conflict)
        insert_data = df_aggregated.to_dict(orient="records")

        if insert_data:
            stmt = insert(XiaomiTemperature30minModele).values(insert_data)
            stmt = stmt.on_conflict_do_update(
                index_elements=["timestamp", "room"],
                set_={
                    "mac_address": stmt.excluded.mac_address,
                    "avg_temperature": stmt.excluded.avg_temperature,
                    "std_temperature": stmt.excluded.std_temperature,
                    "avg_humidity": stmt.excluded.avg_humidity,
                    "std_humidity": stmt.excluded.std_humidity,
                }
            )
            session.execute(stmt)

        # Suppression des données détaillées de plus de 7 jours
        seven_days_ago = datetime.now() - timedelta(days=7)
        session.query(XiaomiTemperature).filter(XiaomiTemperature.timestamp < seven_days_ago).delete(synchronize_session=False)

        session.commit()

    except Exception as e:
        session.rollback()
        print(f"Erreur dans XiaomiTemperature30min : {e}")

    finally:
        session.close()  # Nettoyage de la session après usage

def CozyTouch30min():
    session = SessionSc()  # Récupération de la session isolée
    
    try:
        # Récupérer les données des 6 dernières heures
        six_hours_ago = datetime.now() - timedelta(hours=6)

        recent_entries = (
            session.query(CozyTouchTemperatureModele)
            .filter(CozyTouchTemperatureModele.timestamp >= six_hours_ago)
            .all()
        )

        # Transformer les résultats en DataFrame
        data = [{
            'id': entry.id,
            'piece': entry.piece,
            'timestamp': entry.timestamp,
            'temperature': entry.temperature,
            'temperature_cible': entry.target_temperature,
            'consommation': entry.consumption
        } for entry in recent_entries]

        df = pd.DataFrame(data)

        if df.empty:  # Éviter les erreurs si aucune donnée n'est trouvée
            return  

        df.set_index('timestamp', inplace=True)

        df_aggregated = df.groupby(['piece']).resample('30min').agg(
            moyenne_temperature=('temperature', 'mean'),
            ecart_type_temperature=('temperature', 'std'),
            temperature_cible=('temperature_cible', 'mean'),
            consommation=('consommation', 'max')
        ).reset_index()

        df_aggregated['moyenne_temperature'] = df_aggregated['moyenne_temperature'].round(2)

        # Préparer les données pour un bulk insert avec upsert (on conflict)
        insert_data = df_aggregated.to_dict(orient="records")

        if insert_data:
            stmt = insert(CozyTouchTemperature30minModele).values(insert_data)
            stmt = stmt.on_conflict_do_update(
                index_elements=['timestamp', 'piece'],
                set_={
                    'moyenne_temperature': stmt.excluded.moyenne_temperature,
                    'ecart_type_temperature': stmt.excluded.ecart_type_temperature,
                    'temperature_cible': stmt.excluded.temperature_cible,
                    'consommation': stmt.excluded.consommation,
                }
            )
            session.execute(stmt)

        # Suppression des données détaillées de plus d'une semaine
        one_week_ago = datetime.now() - timedelta(weeks=1)
        session.query(CozyTouchTemperatureModele).filter(
            CozyTouchTemperatureModele.timestamp < one_week_ago
        ).delete(synchronize_session=False)

        session.commit()

    except Exception as e:
        session.rollback()
        print(f"Erreur dans CozyTouch30min : {e}")

    finally:
        session.close()  # Nettoyage de la session après usage


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
    """Met à jour les données de température pour chaque pièce."""
    global temperature_data

    current_date = datetime.now()

    session = SessionSc()

    try:
        # Récupérer toutes les dernières valeurs en une seule requête pour éviter les appels multiples
        latest_records = {
            "Salon": None,
            "Chambre 2": None
        }
        
        for room in latest_records.keys():
            latest_records[room] = (
                session.query(XiaomiTemperature)
                .filter(XiaomiTemperature.room == room)
                .order_by(XiaomiTemperature.timestamp.desc())
                .limit(1)  # Limite à un seul résultat
                .all()
            )

        # Vérifier et mettre à jour les données de température Xiaomi
        for room, record in latest_records.items():
            if record:
                record = record[0]  # Récupérer l'objet unique
                temperature_data[room]["temperature"] = record.temperature
                temperature_data[room]["humidite"] = record.humidity
                temperature_data[room]["timestamp"] = record.timestamp

        # Récupérer la dernière température CozyTouch pour chaque pièce
        latest_records_cozy = session.query(CozyTouchTemperatureModele).filter(
            CozyTouchTemperatureModele.piece.in_(["Salon", "Chambre 2"])
        ).order_by(CozyTouchTemperatureModele.timestamp.desc()).limit(2).all()

        for record in latest_records_cozy:
            temperature_data[record.piece]["temperature cible"] = record.target_temperature

        # Récupérer la température extérieure avec une requête optimisée
        closest_entry = (
            session.query(MeteoAPI30min)
            .order_by(func.abs(func.julianday(MeteoAPI30min.date) - func.julianday(current_date)))
            .limit(1)  # Évite la surcharge mémoire
            .all()
        )

        if closest_entry:
            temperature_data["Exterieur"]["temperature"] = round(closest_entry[0].temperature_2m, 1)

        session.commit()  # Valider les transactions

    except OperationalError as e:
        print(f"Erreur SQLAlchemy: {e}")
        session.rollback()  # Annuler la transaction en cas d'erreur
    finally:
        session.close()  # Libérer les ressources


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
    scheduler.add_job(func=XiaomiTemperature30min, trigger='interval', minutes=20)
    scheduler.add_job(func=CozyTouch30min, trigger='interval', minutes=20)
    scheduler.start()

    app.run(host='0.0.0.0', port=5075, debug=True)