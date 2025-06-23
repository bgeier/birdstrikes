from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from uuid import uuid4

import pandas as pd
from tqdm import tqdm


@dataclass(frozen=True)
class Aircraft:
    atype: str
    ama_code: str
    amo_code: str
    ac_class: str
    ac_mass: int
    num_engs: int
    engine_type: str
    engine_config: str
    registration: str
    faa_region: str


@dataclass
class Engine:
    manufacturer: str
    model: str


@dataclass
class ReportSource:
    person: str
    date: datetime
    lupdated: datetime


@dataclass
class Bird:
    species: str
    size: str
    species_name: str


@dataclass
class Weather:
    cloud_cover: str
    preciptation: str


@dataclass
class Damage:
    severity: str  # substantial, destroyed
    effect_on_flight: str
    damage_locations: list[str]
    struck_locations: list[str]
    ingest_locations: list[str]
    aos_time: int  # days
    repair_cost: int
    repair_cost_adj: int
    other_cost: int
    other_cost_adj: int


@dataclass(frozen=True)
class Airport:
    icao_id: str
    name: str
    state: str
    lat: float
    long: float
    runway: str = "unknown"


@dataclass
class Airline:
    opid: str
    operator: str


@dataclass
class Flight:
    flight_number: str
    flight_type: str


@dataclass
class BirdStrikeEvent:
    remains_collected: bool
    remains_sent: bool
    seen_count: str
    struck_count: str
    phase_of_flight: str
    enroute: str
    height: int  # feet
    speed: int
    distance_from_airport: int  # miles
    pilot_warned: str
    time_of_day: str
    date: datetime
    injury_count: int
    fatality_count: int
    aircraft: Aircraft
    engine: Engine
    report_source: ReportSource
    bird: Bird
    weather: Weather
    damage: Damage
    airport: Airport
    airline: Airline
    flight: Flight
    event_uuid: Optional[str] = field(init=False)

    def __post_init__(self):
        self.event_uuid = str(uuid4())


def parse_birdstrikes(df: pd.DataFrame) -> list[BirdStrikeEvent]:
    birdstrikes: list[BirdStrikeEvent] = []
    for i, row in tqdm(df.iterrows(), desc="Parsing birdstrikes"):
        airport = Airport(
            icao_id=row["AIRPORT_ID"],
            name=row["AIRPORT"],
            state=row["STATE"],
            lat=row["AIRPORT_LATITUDE"],
            long=row["AIRPORT_LONGITUDE"],
        )
        aircraft = Aircraft(
            atype=row["AIRCRAFT"],
            ama_code=row["AMA"],
            amo_code=row["AMO"],
            ac_class=row["AC_CLASS"],
            ac_mass=row["AC_MASS"],
            num_engs=row["NUM_ENGS"],
            engine_type=row["TYPE_ENG"],
            engine_config=f"{row['ENG_1_POS']};{row['ENG_2_POS']};{row['ENG_3_POS']};{row['ENG_4_POS']}",
            registration=row["REG"],
            faa_region=row["FAAREGION"],
        )
        engine = Engine(
            manufacturer=row["EMA"],
            model=row["EMO"],
        )
        report_source = ReportSource(
            person=row["PERSON"],
            date=datetime.fromisoformat(row["INCIDENT_DATE"]),
            lupdated=datetime.fromisoformat(row["LUPDATE"]),
        )
        bird = Bird(
            species=row["SPECIES_ID"],
            size=row["SIZE"],
            species_name=row["SPECIES"],
        )
        weather = Weather(
            cloud_cover=row["SKY"],
            preciptation=row["PRECIPITATION"],
        )
        damage = Damage(
            severity=row["DAMAGE_LEVEL"],
            effect_on_flight=row["EFFECT"],
            damage_locations=[
                f"{col.replace('DAM_', '')};"
                for col in row.index
                if col.startswith("DAM_") and row[col] == 1
            ],
            struck_locations=[
                f"{col.replace('STR_', '')};"
                for col in row.index
                if col.startswith("STR_") and row[col] == 1
            ],
            ingest_locations=[
                f"{col.replace('ING_', '')};"
                for col in row.index
                if col.startswith("ING_") and row[col] == 1
            ],
            aos_time=row["AOS"],
            repair_cost=row["COST_REPAIRS"],
            repair_cost_adj=row["COST_REPAIRS_INFL_ADJ"],
            other_cost=row["COST_OTHER"],
            other_cost_adj=row["COST_OTHER_INFL_ADJ"],
        )
        airline = Airline(
            opid=row["OPID"],
            operator=row["OPERATOR"],
        )
        flight = Flight(
            flight_number=row["FLT"],
            flight_type="COM"
            if row["OPID"] not in ["PVT", "UNK", "BUS", "MIL", "GOV", "FDX", "UPS"]
            else row["OPID"],
        )
        birdstrike_event = BirdStrikeEvent(
            remains_collected=row["REMAINS_COLLECTED"] == 1,
            remains_sent=row["REMAINS_SENT"] == 1,
            seen_count=row["NUM_SEEN"],
            struck_count=row["NUM_STRUCK"],
            phase_of_flight=row["PHASE_OF_FLIGHT"],
            enroute=row["ENROUTE_STATE"],
            height=row["HEIGHT"],
            speed=row["SPEED"],
            distance_from_airport=row["DISTANCE"],
            pilot_warned=row["WARNED"],
            time_of_day=row["TIME_OF_DAY"],
            date=datetime.fromisoformat(row["INCIDENT_DATE"]),
            injury_count=row["NR_INJURIES"],
            fatality_count=row["NR_FATALITIES"],
            aircraft=aircraft,
            engine=engine,
            report_source=report_source,
            bird=bird,
            weather=weather,
            damage=damage,
            airport=airport,
            airline=airline,
            flight=flight,
        )
        birdstrikes.append(birdstrike_event)

    return birdstrikes
