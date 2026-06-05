categorical_features: list = [
    "STATE",
    "SA4",
    "SA3",
    "POSTCODE",
    "SUBURB",
    "SA2",
    "SA1",
    "ZONING",
    # "HIERARCHY",
    # "LOCALITY_ID",
    "STREET_TYPE",  # add in to see impact
    "PROPERTY_TYPE",
    "PROPERTY_SUBTYPE",
]
numerical_features: list = [
    "YEAR_BUILT",
    # "LATITUDE",
    # "LONGITUDE",
    "BEDROOM",
    "BATHROOM",
    "CARSPACE",
    "LAND_AREA",
    "INTERNAL_AREA",
    "PREVIOUS_SOLD_PRICE",  # T 4.8/13 is null
    "FROM_PS",
    "SOLD_PRICE",
    "YEAR_BUILT",
    "DAYS_ON_MARKET",
]

training_features_num = [
    "YEAR_BUILT",
    "BEDROOM",
    "BATHROOM",
    "CARSPACE",
    "LAND_AREA",
    "INTERNAL_AREA",
    "PREVIOUS_SOLD_PRICE",
    "FROM_PS",
    "SOLD_PRICE",
    "DAYS_ON_MARKET",
]

training_features_cat = ["PROPERTY_SUBTYPE"]

features_to_impute = [
    "bedroom",
    "bathroom",
    "carspace",
    "land_area",
    "internal_area",
]
