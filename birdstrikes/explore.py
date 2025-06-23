import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go  # or plotly.express as px
import typer
from dash import Dash, Input, Output, callback, dcc, html
from plotly.subplots import make_subplots
from tqdm import tqdm

from .entity_models import (
    Aircraft,
    Airport,
    Bird,
    BirdStrikeEvent,
    Damage,
    Engine,
    parse_birdstrikes,
)

tapp = typer.Typer()


def convert_coords(map_content) -> list[str] | None:
    return (
        [
            stringify_coords(point["lat"], point["lon"])
            for point in map_content["points"]
        ]
        if map_content
        else None
    )


def filter_strikes(
    strikes: list[BirdStrikeEvent],
    start_date: str,
    end_date: str,
    aircraft: str | None = None,
    engine: str | None = None,
    damage: str | None = None,
    bird: str | None = None,
    coords: list[str] | None = None,
) -> None | list[BirdStrikeEvent]:
    start_date = datetime.fromisoformat(start_date)
    end_date = datetime.fromisoformat(end_date)
    # filter by date
    strikes = [
        strike
        for strike in strikes
        if strike.date >= start_date and strike.date <= end_date
    ]

    # filter by aircraft
    if aircraft and aircraft != "all":
        strikes = [strike for strike in strikes if strike.aircraft.atype == aircraft]

    # filter by engine
    if engine and engine != "all":
        strikes = [
            strike for strike in strikes if strike.aircraft.engine_type == engine
        ]

    # filter by bird
    if bird and bird != "all":
        strikes = [strike for strike in strikes if strike.bird.species == bird]

    if damage and damage != "all":
        strikes = [strike for strike in strikes if strike.damage.severity == damage]

    if coords:
        strikes = [
            strike
            for strike in strikes
            if stringify_coords(strike.airport.lat, strike.airport.long) in coords
        ]

    return strikes if strikes else None


def bird_df(birds: list[Bird]) -> pd.DataFrame:
    species_count = defaultdict(int)
    species_id_to_label = {}
    for bird in birds:
        species_count[bird.species] += 1
        if bird.species not in species_id_to_label:
            species_id_to_label[bird.species] = bird.species_name

    _bird_df = pd.DataFrame.from_dict(data=species_count, orient="index")
    _bird_df.columns = ["cnt"]
    _bird_df["species"] = _bird_df.index
    _bird_df["species_label"] = [
        species_id_to_label[species_id] for species_id in _bird_df["species"]
    ]
    return _bird_df


def airport_df(airports: list[Airport]) -> pd.DataFrame:
    # return a dataframe with counts by unique airports
    # lat long name count
    airport_count = defaultdict(int)
    for airport in airports:
        try:
            if math.isnan(airport.lat):
                continue
        except TypeError:
            print(airport)
            continue
        airport_count[(airport.lat, airport.long)] += 1
    agg_df = pd.DataFrame.from_dict(data=airport_count, orient="index")
    agg_df.columns = ["cnt"]
    agg_df["cnt"] = [math.log2(val) for val in agg_df["cnt"]]
    agg_df["lat"] = [val[0] for val in agg_df.index]
    agg_df["long"] = [val[1] for val in agg_df.index]
    return agg_df


def aircraft_df(aircrafts: list[Aircraft]) -> tuple[pd.DataFrame, pd.DataFrame]:
    aircraft_count = defaultdict(int)
    ac_mass_count = defaultdict(int)
    for aircraft in aircrafts:
        aircraft_count[aircraft.atype] += 1
        ac_mass_count[aircraft.ac_mass] += 1
    agg_df = pd.DataFrame.from_dict(data=aircraft_count, orient="index")
    agg_df.columns = ["cnt"]
    # agg_df["cnt"] = [math.log2(val) for val in agg_df["cnt"]]
    agg_df["atype"] = agg_df.index

    mass_df = pd.DataFrame.from_dict(data=ac_mass_count, orient="index")
    mass_df.columns = ["cnt"]
    # agg_df["cnt"] = [math.log2(val) for val in agg_df["cnt"]]
    mass_df["ac_class"] = mass_df.index

    label_dict = {
        1: "2,250 kg or less",
        2: "2,250 - 5,700 kg",
        3: "5,701 - 27,000 kg",
        4: "27,001 - 272,000 kg",
        5: "Above 272,000 kg",
    }
    mass_df["ac_class_label"] = [
        label_dict[val] if val in label_dict else val for val in mass_df["ac_class"]
    ]

    return agg_df, mass_df


def damage_df(
    damages: list[Damage],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    severity_count = defaultdict(int)
    eof_count = defaultdict(int)
    struck = defaultdict(int)
    ingest = defaultdict(int)
    for damage in damages:
        severity_count[damage.severity] += 1
        eof_count[damage.effect_on_flight] += 1
        damage_label = f"{''.join(damage.struck_locations)}"
        damage_label = "none reported" if damage_label == "" else damage_label
        struck[damage_label] += 1
        damage_label = f"{''.join(damage.ingest_locations)}"
        damage_label = "none reported" if damage_label == "" else damage_label
        ingest[damage_label] += 1

    sdf = pd.DataFrame.from_dict(data=severity_count, orient="index")
    sdf.columns = ["cnt"]
    sdf["severity"] = sdf.index
    label_dict = {
        "N": "None",
        "M": "Minor",
        "M?": "Undetermined",
        "S": "Substantial",
        "D": "Destroyed",
    }
    sdf["severity_label"] = [
        label_dict[slevel] if slevel in label_dict else slevel
        for slevel in sdf["severity"]
    ]

    edf = pd.DataFrame.from_dict(data=eof_count, orient="index")
    edf.columns = ["cnt"]
    edf["effect_on_flight"] = edf.index

    dcdf = pd.DataFrame.from_dict(data=struck, orient="index")
    dcdf.columns = ["cnt"]
    dcdf["damage"] = dcdf.index
    idf = pd.DataFrame.from_dict(data=ingest, orient="index")
    idf.columns = ["cnt"]
    idf["damage"] = idf.index
    return sdf, edf, dcdf, idf


def time_df(
    strikes: list[BirdStrikeEvent],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    year_cnts = defaultdict(int)
    month_cnts = defaultdict(int)
    tod_cnts = defaultdict(int)
    for strike in strikes:
        year, month = strike.date.year, strike.date.month
        year_cnts[year] += 1
        month_cnts[month] += 1
        tod_cnts[strike.time_of_day] += 1

    year_df = pd.DataFrame.from_dict(data=year_cnts, orient="index")
    year_df.columns = ["cnt"]
    year_df["year"] = year_df.index

    month_df = pd.DataFrame.from_dict(data=month_cnts, orient="index")
    month_df.columns = ["cnt"]
    month_df["month"] = month_df.index

    agg_df = pd.DataFrame.from_dict(data=tod_cnts, orient="index")
    agg_df.columns = ["cnt"]
    agg_df["TOD"] = agg_df.index
    return year_df, month_df, agg_df


def create_map_figure(df):
    fig = px.scatter_map(
        df,
        lat="lat",
        lon="long",
        color="cnt",
        size="cnt",
        color_continuous_scale=px.colors.cyclical.IceFire,
        size_max=df["cnt"].max() * 0.9,
        labels={"cnt": "log2 Event Count"},
        zoom=2,
    )
    fig.update_layout(width=1400, height=600)
    return fig


def stringify_coords(lat: float, long: float) -> str:
    return f"({lat},{long})"


@tapp.command()
def main(file: Path):
    print("reading data")
    if file.suffix == ".csv":
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file, sheet_name="Sheet1")

    strikes = parse_birdstrikes(df)

    app = Dash()
    app.layout = html.Div(
        children=[
            html.H1("Explore FAA Bird Strike Dataset"),
            html.Div(html.H2("Available Data Filters")),
            html.Div(
                html.H3("Select a date range to explore:"),
                style={"width": "20%", "display": "inline-block"},
            ),
            html.Div(
                dcc.DatePickerRange(
                    id="date-picker-range",
                    min_date_allowed=df["INCIDENT_DATE"].min(),
                    max_date_allowed=df["INCIDENT_DATE"].max(),
                    initial_visible_month=df["INCIDENT_DATE"].min(),
                    end_date=df["INCIDENT_DATE"].max(),
                    start_date=df["INCIDENT_DATE"].min(),
                ),
                style={"width": "80%", "display": "inline-block"},
            ),
            html.Div(
                html.H3("Select an aircraft to display:"),
                style={"width": "20%", "display": "inline-block"},
            ),
            html.Div(
                dcc.Dropdown(
                    options=list(df["AIRCRAFT"].unique()) + ["all"],
                    value="all",
                    id="ac-type-select",
                ),
                style={"width": "80%", "display": "inline-block"},
            ),
            dcc.Graph(id="map-content"),
            html.Div(
                html.H4(
                    "Note: Use map selection tools (e.g., lasso select) to focus analysis on a geographic area of interest"
                )
            ),
            html.Div(html.H2("Time Effects within Filter Criteria")),
            dcc.Graph(id="time-marginals"),
            html.Div(
                html.H2(
                    "Distribution Analysis of Event Variables within Applied Filters and or Map Selection"
                )
            ),
            html.Div(
                dcc.Graph(id="aircraft-type"),
                style={"width": "49%", "display": "inline-block"},
            ),
            html.Div(
                dcc.Graph(id="ac-class-fig"),
                style={"width": "49%", "display": "inline-block"},
            ),
            html.Div(
                dcc.Graph(id="damage"),
                style={"width": "60%", "display": "inline-block"},
            ),
            html.Div(
                dcc.Graph(id="ingest"),
                style={"width": "40%", "display": "inline-block"},
            ),
            html.Div(
                dcc.Graph(id="bird-overall"),
                style={"width": "49%", "display": "inline-block"},
            ),
            html.Div(
                dcc.Graph(id="bird-selected"),
                style={"width": "49%", "display": "inline-block"},
            ),
            html.Div(
                dcc.Graph(id="damage-eof"),
                style={"width": "49%", "display": "inline-block"},
            ),
            html.Div(
                dcc.Graph(id="damage-severity"),
                style={"width": "49%", "display": "inline-block"},
            ),
        ]
    )

    # bird species
    @callback(
        Output("bird-overall", "figure"),
        Output("bird-selected", "figure"),
        Input("date-picker-range", "start_date"),
        Input("date-picker-range", "end_date"),
        Input("map-content", "selectedData"),
        Input("ac-type-select", "value"),
    )
    def update_birds(start_date, end_date, map_content, aircraft):
        coords = convert_coords(map_content=map_content)

        bird_df_selected = bird_df(
            [
                strike.bird
                for strike in filter_strikes(
                    strikes,
                    start_date=start_date,
                    end_date=end_date,
                    coords=coords,
                    aircraft=aircraft,
                )
            ]
        )
        bird_df_overall = bird_df(
            [
                strike.bird
                for strike in filter_strikes(
                    strikes,
                    start_date=start_date,
                    end_date=end_date,
                    aircraft=aircraft,
                )
            ]
        )
        fig_overall = px.pie(
            bird_df_overall,
            values="cnt",
            names="species",
            title="Overall Distribution of Bird Species",
            labels={"cnt": "Event Count", "species": "Species ID"},
            hover_name="species_label",
        )
        fig_overall.update_traces(textposition="inside", textinfo="percent+label")
        fig_filtered = px.pie(
            bird_df_selected,
            values="cnt",
            names="species",
            title="Distribution of Bird Species within Map Selection",
            labels={"cnt": "Event Count", "species": "Species ID"},
            hover_name="species_label",
        )
        fig_filtered.update_traces(textposition="inside", textinfo="percent+label")
        return fig_overall, fig_filtered

    @callback(
        Output("damage-severity", "figure"),
        Output("damage-eof", "figure"),
        Output("damage", "figure"),
        Output("ingest", "figure"),
        Input("date-picker-range", "start_date"),
        Input("date-picker-range", "end_date"),
        Input("map-content", "selectedData"),
        Input("ac-type-select", "value"),
    )
    def update_damage_severity(start_date, end_date, map_content, aircraft):
        coords = convert_coords(map_content=map_content)
        strike_events = filter_strikes(
            strikes,
            start_date=start_date,
            end_date=end_date,
            coords=coords,
            aircraft=aircraft,
        )
        severity, eof, struck_damage, ingest_damage = damage_df(
            damages=[strike.damage for strike in strike_events]
        )
        fig_ds = px.pie(
            severity,
            values="cnt",
            names="severity_label",
            title="Distribution of Damage Severity",
        )
        fig_ds.update_traces(textposition="inside", textinfo="percent+label")

        fig_eof = px.pie(
            eof,
            values="cnt",
            names="effect_on_flight",
            title="Distribution of Damage Effect on Flight",
        )
        fig_eof.update_traces(textposition="inside", textinfo="percent+label")

        fig_damage = px.pie(
            struck_damage,
            values="cnt",
            names="damage",
            title="Distribution of Struck Damage",
        )
        fig_damage.update_traces(textposition="inside", textinfo="percent+label")

        fig_ingest = px.pie(
            ingest_damage,
            values="cnt",
            names="damage",
            title="Distribution of Ingest Damage",
        )
        fig_ingest.update_traces(textposition="inside", textinfo="percent+label")

        return fig_ds, fig_eof, fig_damage, fig_ingest

    @callback(
        Output("map-content", "figure"),
        Input("date-picker-range", "start_date"),
        Input("date-picker-range", "end_date"),
        Input("ac-type-select", "value"),
    )
    def update_map(start_date, end_date, aircraft):  # , engine, damage, bird):
        strike_events = filter_strikes(
            strikes,
            start_date=start_date,
            end_date=end_date,
            aircraft=aircraft,
        )

        adf = airport_df([strike.airport for strike in strike_events])

        return create_map_figure(adf)

    @callback(
        Output("time-marginals", "figure"),
        Input("date-picker-range", "start_date"),
        Input("date-picker-range", "end_date"),
        Input("map-content", "selectedData"),
        Input("ac-type-select", "value"),
    )
    def update_time(start_date, end_date, map_content, aircraft):
        event_strikes = filter_strikes(
            strikes, start_date=start_date, end_date=end_date, aircraft=aircraft
        )

        if map_content:
            coords = [
                stringify_coords(point["lat"], point["lon"])
                for point in map_content["points"]
            ]

            year_df_s, month_df_s, tod_df_s = time_df(
                [
                    strike
                    for strike in event_strikes
                    if stringify_coords(
                        strike.airport.lat,
                        strike.airport.long,
                    )
                    in coords
                ]
            )
        else:
            year_df_s, month_df_s, tod_df_s = None, None, None
        year_df, month_df, tod_df = time_df(strikes)
        fig = make_subplots(
            rows=1 if year_df_s is None else 2,
            cols=3,
            subplot_titles=[
                "Overall Year Event Distribution",
                "Overall Month Event Distribution",
                "Overall Time of Day Distribution",
            ]
            if year_df_s is None
            else [
                "Overall Year Event Distribution",
                "Overall Month Event Distribution",
                "Overall Time of Day Distribution",
                "Year Event Distribution within Map Select",
                "Month Event Distribution within Map Select",
                "Time of Day Distribution within Map Select",
            ],
        )
        fig.add_trace(
            go.Bar(name="foo", x=year_df["year"], y=year_df["cnt"]),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(x=month_df["month"], y=month_df["cnt"]),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Bar(
                x=tod_df["TOD"],
                y=tod_df["cnt"],
            ),
            row=1,
            col=3,
        )
        if year_df_s is not None:
            fig.add_trace(
                go.Bar(name="foo", x=year_df_s["year"], y=year_df_s["cnt"]),
                row=2,
                col=1,
            )

            fig.add_trace(
                go.Bar(x=month_df_s["month"], y=month_df_s["cnt"]),
                row=2,
                col=2,
            )
            fig.add_trace(
                go.Bar(
                    x=tod_df_s["TOD"],
                    y=tod_df_s["cnt"],
                ),
                row=2,
                col=3,
            )

        fig.update_layout(showlegend=False)
        return fig

    @callback(
        Output("aircraft-type", "figure"),
        Output("ac-class-fig", "figure"),
        Input("date-picker-range", "start_date"),
        Input("date-picker-range", "end_date"),
        Input("map-content", "selectedData"),
    )
    def update_aircraft(start_date, end_date, map_content):
        coords = convert_coords(map_content=map_content)
        strike_events = filter_strikes(
            strikes,
            start_date=start_date,
            end_date=end_date,
            coords=coords,
        )
        adf, mass_df = aircraft_df([strike.aircraft for strike in strike_events])

        fig = px.pie(
            adf,
            values="cnt",
            names="atype",
            title="Distribution of Aircraft Types",
            labels={"cnt": "Bird Strike Count", "atype": "Aircraft Type"},
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")

        fig2 = px.pie(
            mass_df,
            values="cnt",
            names="ac_class_label",
            title="Distribution of AC Mass",
            labels={"cnt": "Bird Strike Count", "ac_class_label": "AC Class"},
        )
        fig2.update_traces(textposition="inside", textinfo="percent+label")

        return fig, fig2

    app.run(debug=True, use_reloader=True)  # Turn off reloader if inside Jupyter


if __name__ == "__main__":
    tapp()
