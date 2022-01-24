class config():
    Skip_Words = {"the", "and", "a", "of", "to", "is", "it", "in", "i", "this", "that", "was", "as", "for", "with",
                  "but", "on", "not", "you", "he", "are", "his", "have", "be",
                  "best","life","hope","family","life","catch","class","self","bear","case","free","balance","active","living",
                  "married", "dead","anti",'modern', 'double'}

    attributes_key = {"person", "human", "people", "human adult", "adult", "person_with_nationality","employee"}


    sex_pair_set = {"brother", "sister", "aunt", "uncle", "waiter", "waitress", "father", "mother", "policeman",
                    "policewoman", "son", "daughter", "gentlemen", "lady", "host", "hostess", "grandfather", "grandmother",
                    "actor", "actress", "prince", "princess", "boy", "girl","men","women","man","woman"}

    sex_pair = {"brother":"sister","sister":"brother","aunt":"uncle","uncle":"aunt","waiter":"waitress","waitress":"waiter"
                ,"father":"mother","mother":"father","policeman":"policewoman","policewoman":"policeman","son":"daughter","daughter":"son"
                ,"gentlemen":"lady","lady":"gentlemen","host":"hostess","hostess":"host","grandfather":"grandmother","grandmother":"grandfather"
                ,"actor":"actress","actress":"actor","prince":"princess","princess":"prince","boy":"girl","girl":"boy","men":"women",
                "man":"woman","women":"men","woman":"man" }

    country_set = {'tajik', 'portuguese','argentinian', 'sicilian', 'moroccan','philippino',
                     'pakistani', 'west_indian', 'indian', 'ivorian', 'zambian',  'israeli', 'gambian',
                     'lebanese_person', 'romanian', 'venezuelan', 'dominican', 'new_zealander', 'mexican', 'sudanese',
                     'african', 'saint_lucian', 'papua_new_guinean', 'yemini_american', 'bahraini',  'chilean', 'malawian',
                     'burmese', 'senegalese',  'nauruan', 'belizean', 'dane', 'kenyan', 'tunisian', 'zairian','grenadan',
                     'chinese', 'bulgarian', 'mauritian', 'norwegian', 'magyar',  'spaniard','maldive', 'gibraltarian',
                     'french_polynesian', 'djiboutian', 'bruneian', 'mahorais', 'emirati', 'armenian', 'yugoslav',
                      'liechtensteiner', 'morrocan', 'guyanese', 'scot', 'iranian', 'afghan', 'gabonese', 'taiwanese',
                     'south_african', 'nipponese', 'togolese', 'north_american', 'australian','corsican',
                     'slovenian', 'ethiopian', 'libyan',  'guatemalan', 'german', 'burkina_fasoan', 'mozambican',
                     'kittitian', 'reunionese', 'russian',  'salvadorian', 'cypriot', 'eritrean', 'kuwaiti',
                     'west_german', 'ukrainian', 'mauritanian', 'vanuatuan', 'singaporean', 'chadian', 'latvian', 'south_korean', 'dutchman',
                     'guinean', 'lao', 'cabindan', 'belgian', 'ghanese', 'andorran', 'rio_munian', 'liberian',
                     'italian', 'yemeni', 'yemen', 'englishman',  'sikkim', 'western_samoan', 'haitian',
                     'canadian', 'syriac', 'hong_kong', 'czech', 'israeli', 'basotho', 'madagascan', 'botswanian',
                     'macedonian', 'iraqi', 'bolivian', 'uruguayan','vietnamese', 'sri_lankan', 'siamese', 'albanian',
                     'finn', 'batik', 'turk', 'slovak', 'austrian', 'swedish', 'guinea_bissauan', 'cuban',  'congolese',
                      'nepalese', 'rwandan', 'tuvaluan', 'palestinian', 'yemeni', 'czechoslovakian',  'monacan',
                     'croatian', 'tanzanian', 'mongolian', 'bangladeshi', 'sao_tomean', 'british', 'phillipino', 'algerian', 'nicaraguan',
                     'palestine', 'russian', 'sardinian', 'beninese',  'brazilian', 'icelandic', 'swiss', 'irish','estonian',
                     'asian', 'qatari', 'iraqi', 'manx', 'greek', 'malaysian', 'azerbaijani', 'kiribati', 'paraguayan', 'malian',
                     'japanese', 'omani', 'new_caledonian', 'ugandan', 'kyrgyzstani', 'american', 'zimbabwean','san_marinese',
                     'bermudian',  'lithuanian', 'puerto_rican','seychellean', 'peruvian', 'texan', 'cameroonian',
                      'turkmen', 'nigerian', 'georgian', 'angolan', 'surinamese',  'swazi', 'tongan',  'panamanian',
                     'luxemburger', 'central_american', 'nigerois', 'saint_vincentian','namibian', 'soviet','welshman', 'maltese',
                     'belarussian', 'bahaman', 'burundian','european', 'indonesian', 'jamaican', 'honduran', 'belarusian', 'saudi',
                     'cambodian', 'somali',  'south_american',  'ecuadoran', 'colombian', 'guinean', 'barbadian', 'fijian', 'east_german', }

    country_set2 = {'north_korean_person','comoran_person','emirati_person','brechou_island_person', 'okinawa_islands_person',
                    'volcano_islands_person','azerbaijani_person','jordanian_person','herm_island_person','jethou_person','great_sark_person',
                    'northern_irish_person','daito_islands_person','cape_verdian_person', 'melilla_enclave_person','lithou_island_person',
                    'cook_islands_person', 'faeroe_islands_person','biffeche_person', 'jersey_island_person', 'french_person','canadian_person',
                    'united_states_citizen', 'tokelau_islands_person',  'saint_helena_island_person', 'netherlands_antilles_person',
                    'citizen_of_puerto_alvira', 'alderney_island_person', 'cocos_islands_person', 'marcus_islands_person','greenland_person',
                    'ceuta_enclave_person', 'canary_island_person', 'sierra_leonean_person', 'scandinavian_person','british_virgin_islands_person',
                    'guernsey_island_person','pitcairn_island_person','falkland_islands_dependencies_person',  'polish_person',
                    'martinique_person', 'south_georgia_island_person', 'penghu_islands_person','marshall_islands_person','columbia_citizen',
                    'azores_islands_person', 'iranian_person', 'falkland_islands_person','ascension_island_person', 'macao_enclave_person',
                    'egyptian_person', 'little_sark_island_person', 'antiguan_person', 'niuean_person', 'trinidad_and_tobagan','vatican_city_citizen',
                    'saudi_citizen','ancient_roman', 'iraqi_citizen','guadeloupe_person', 'costa_rican', 'madeira_islander','solomon_islands',
                    'norfolk_islander', 'svalbard_islands_person', 'sierra_leonean',}

if __name__=="__main__":
    sex_pair = {"brother":"sister",}

    print(sex_pair["sister"])