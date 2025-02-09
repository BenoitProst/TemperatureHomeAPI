import requests, shelve, json, time, unicodedata, os, sys, errno
from datetime import datetime
import pandas as pd
from openpyxl import load_workbook

from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String, Float, ForeignKey, Table, DateTime, UniqueConstraint
from sqlalchemy.dialects.sqlite import insert

import json

engine = create_engine('sqlite:////home/pi/CozyTouchAPI/data/Thermometer.db')

# Définir une base pour les modèles
Base = declarative_base()

url_cozytouchlog=u'https://ha110-1.overkiz.com/enduser-mobile-web/enduserAPI/'
url_cozytouch=u'https://ha110-1.overkiz.com/enduser-mobile-web/externalAPI/json/'

url_atlantic=u'https://apis.groupe-atlantic.com'

cozytouch_save = '/home/pi/CozyTouchAPI'+'/cozytouch_save'


debug=1 # 0 : pas de traces debug / 1 : traces requêtes http / 2 : dump data json reçues du serveur cozytouch / 4 : dump data device sauvegardés / 555 : pour lier manuellement les devices déjà existants en cas de suppresion malencontreuse du cozytouch_save mais pas des devices

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

def var_save(var, var_str):
    """Fonction de sauvegarde
    var: valeur à sauvegarder, var_str: nom objet en mémoire
    """
    d = shelve.open(cozytouch_save)
    if var_str in d :
        d[var_str] = var

    else :
        d[var_str] = 0 # init variable
        d[var_str] = var

    d.close()

def var_restore(var_str,format_str =False ):
    '''Fonction de restauration
    var_str: nom objet en mémoire
    '''
    d = shelve.open(cozytouch_save)
    if not (var_str) in d :
        if  format_str:
            value = 'init' # init variable
        else :
            value = 0 # init variable
    else :
        value = d[var_str]
    d.close()
    return value

def http_error(code_erreur, texte_erreur):
    ''' Evaluation des exceptions HTTP '''
    print("Erreur HTTP "+str(code_erreur)+" : "+texte_erreur)


def cozytouch_login(login,password):


    headers={
    'Content-Type':'application/x-www-form-urlencoded',
    'Authorization':'Basic Q3RfMUpWeVRtSUxYOEllZkE3YVVOQmpGblpVYToyRWNORHpfZHkzNDJVSnFvMlo3cFNKTnZVdjBh'
        }
    data={
        'grant_type':'password',
        'username':'GA-PRIVATEPERSON/' + login,
        'password':password
        }

    url=url_atlantic+'/token'
    req = requests.post(url,data=data,headers=headers)

    atlantic_token=req.json()['access_token']

    headers={
    'Authorization':'Bearer '+atlantic_token+''
        }
    reqjwt=requests.get(url_atlantic+'/magellan/accounts/jwt',headers=headers)

    print(reqjwt.content)

    jwt=reqjwt.content.decode('utf-8').replace('"','')
    data={
        'jwt':jwt
        }
    jsession=requests.post(url_cozytouchlog+'login',data=data)

    if debug:
        print(' POST-> '+url_cozytouchlog+"login | userId=****&userPassword=**** : "+str(jsession.status_code))

    if jsession.status_code==200 : # Réponse HTTP 200 : OK
        print("Authentification serveur cozytouch OK")
        cookies =dict(JSESSIONID=(jsession.cookies['JSESSIONID'])) # Récupération cookie ID de session
        var_save(cookies,'cookies') #Sauvegarde cookie
        return True

    print("!!!! Echec authentification serveur cozytouch")
    http_error(req.status_code,req.reason)
    return False


def cozytouch_GET(json):
    ''' Fonction d'interrogation HTTP GET avec l'url par défaut
    json: nom de fonction JSON à transmettre au serveur
    '''
    headers = {
    'cache-control': "no-cache",
    'Host' : "ha110-1.overkiz.com",
    'Connection':"Keep-Alive",
    }
    myurl=url_cozytouchlog+json
    cookies=var_restore('cookies')
    req = requests.get(myurl,headers=headers,cookies=cookies)
    if debug:
        print(u'  '.join((u'GET-> ',myurl,' : ',str(req.status_code))).encode('utf-8'))

    if req.status_code==200 : # Réponse HTTP 200 : OK
            data=req.json()
            return data

    http_error(req.status_code,req.reason) # Appel fonction sur erreur HTTP
    time.sleep(1) # Tempo entre requetes
    return None


# Ouvrir et lire le fichier JSON
with open("/home/pi/CozyTouchAPI/param/param.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Accéder aux valeurs
login = data["login"]
password = data["password"]


# login = 'benoitprost@yahoo.fr'


cozytouch_login(login,password)

data = cozytouch_GET('setup')


TemperatureSensorsUrls = {
    'Salon': 'io://0836-0474-9942/14618354#2',
    'Entrée': 'io://0836-0474-9942/1292639#2',
    'Chambre 1': 'io://0836-0474-9942/12427457#2',
    'Chambre 2': 'io://0836-0474-9942/9316128#2'
}

TemperatureTargetsUrls = {
    'Salon': 'io://0836-0474-9942/14618354#1',
    'Entrée': 'io://0836-0474-9942/1292639#1',
    'Chambre 1': 'io://0836-0474-9942/12427457#1',
    'Chambre 2': 'io://0836-0474-9942/9316128#1'
}

CumulatedElectricalEnergyConsumptionsUrls = {
    'Salon': 'io://0836-0474-9942/14618354#5',
    'Entrée': 'io://0836-0474-9942/1292639#5',
    'Chambre 1': 'io://0836-0474-9942/12427457#5',
    'Chambre 2': 'io://0836-0474-9942/9316128#5'
}

# Listes pour stocker les données vins
piecelist = []
timestamps = []
temperature = []
consommation = []
temperature_cible = []

# Récupération de la température relevée
for piece, url in TemperatureSensorsUrls.items():
  current_time = datetime.now()
  timestamps.append(current_time)
  piecelist.append(piece)

  target_device_url = url

  # Rechercher le bloc correspondant
  result = next((item for item in data['devices'] if item.get('deviceURL') == target_device_url), None)

  # Afficher le résultat
  if result:

      # Trouver la valeur de core:TemperatureState
      temperature_value = next((state['value'] for state in result['states'] if state['name'] == 'core:TemperatureState'), None)

      if temperature_value is not None:
          temperature.append(temperature_value)


#Récupération de la consommation cummulée
for piece, url in CumulatedElectricalEnergyConsumptionsUrls.items():
  target_device_url = url

  # Rechercher le bloc correspondant
  result = next((item for item in data['devices'] if item.get('deviceURL') == target_device_url), None)

  if result:

        # Trouver la valeur de core:TemperatureState
        consumption_value = next((state['value'] for state in result['states'] if state['name'] == 'core:ElectricEnergyConsumptionState'), None)

        if consumption_value is not None:
          consommation.append(consumption_value)


#Récupération de la température cible
for piece, url in TemperatureTargetsUrls.items():
  target_device_url = url
  # Rechercher le bloc correspondant
  result = next((item for item in data['devices'] if item.get('deviceURL') == target_device_url), None)

  if result:

    temperature_setpoint = None
    for state in result['states']:
      if state['name'] == 'io:EffectiveTemperatureSetpointState':
        temperature_setpoint = state['value']
        break
  temperature_cible.append(temperature_setpoint)


CozyTouch_Informations = {
  'timestamp': timestamps,
  'piece': piecelist,
  'temperature': temperature,
  'temperature cible': temperature_cible,
  'consommation': consommation
}

dfCozyTouchInfo = pd.DataFrame(CozyTouch_Informations)

# Boucle pour insérer chaque ligne
for _, row in dfCozyTouchInfo.iterrows():
    stmt = insert(CozyTouchTemperatureModele).values(
        timestamp=row['timestamp'],
        piece=row['piece'],
        temperature=row['temperature'],
        target_temperature=row['temperature cible'],
        consumption=row['consommation']
    )

    # En cas de conflit, ignorer l'insertion
    stmt = stmt.on_conflict_do_nothing(
        index_elements=['timestamp', 'piece']  # Clés utilisées pour détecter le conflit
    )

    session.execute(stmt)

# Commit pour sauvegarder dans la base de données
session.commit()

#Initialisation des sauvegardes csv et xls

# dfCozyTouchInfo.to_csv('/home/pi/CozyTouchAPI/CozyTouch_data.csv', index=False)

#Append des sauvegardes csv et xls

# dfCozyTouchInfo.to_csv('/home/pi/CozyTouchAPI/data/CozyTouch_data.csv', mode='a', header=False, index=False)

