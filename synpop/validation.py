

def fix_mobility(persons):
    """After any changes in one of the boolean mobility variables, the 'mobility' variable itself must be fixed."""
    # fix Abos
    persons['has_va'] = persons['has_va'].mask(persons['has_ga'], False)
    persons['has_ht'] = persons['has_ht'].mask(persons['has_ga'], False)
    # fix mobility for car_available
    persons['mobility'] = persons['mobility'].mask(persons['car_available'], 'car')
    # fix mobility for GA
    persons['mobility'] = persons['mobility'].mask(persons['has_ga'], 'ga')
    persons['mobility'] = persons['mobility'].mask(
        persons['has_ga'] & persons['car_available'], 'car & ga')
    # fix mobility for VA
    persons['mobility'] = persons['mobility'].mask(persons['has_va'], 'va')
    persons['mobility'] = persons['mobility'].mask(
        persons['has_va'] & persons['car_available'], 'car & va')
    # fix mobility for HT
    persons['mobility'] = persons['mobility'].mask(persons['has_ht'], 'ht')
    persons['mobility'] = persons['mobility'].mask(
        persons['has_ht'] & persons['car_available'], 'car & ht')
    persons['mobility'] = persons['mobility'].mask(
        persons['has_va'] & persons['has_ht'], 'va & ht')
    persons['mobility'] = persons['mobility'].mask(
        persons['has_va'] & persons['has_ht'] & persons['car_available'], 'car & va & ht')
    # fix mobility for nothing
    persons['mobility'] = persons['mobility'].mask(
        ~persons['has_ga'] & ~persons['has_va'] & ~persons['has_ht'] & ~persons['car_available'], 'nothing')
    return persons


def validate_person_groups(persons):
    """These are the person groups used in mobi-plans. This tests that no group is overlapping."""

    def validate_person_groups(person_groups):
        indexes = {}
        for pg, query in person_groups.items():
            indexes[pg] = persons.query(query).index
        for pg, index in indexes.items():
            for pg2, index2 in indexes.items():
                if pg != pg2:
                    assert ~index.isin(index2).any(), f"{pg} and {pg2} have overlapping persons!"

    empl_queries = {
        'Empl_GA_Car': 'current_job_rank != "null" & car_available & has_ga',
        'Empl_GA_NoCar': 'current_job_rank != "null" & ~car_available & has_ga',
        'Empl_VA_Car': 'current_job_rank != "null" & car_available & has_va',
        'Empl_VA_NoCar': 'current_job_rank != "null" & ~car_available & has_va',
        'Empl_NoAbo_Car': 'current_job_rank != "null" & car_available & ~has_ga & ~has_va',
        'Empl_NoAbo_NoCar': 'current_job_rank != "null" & ~car_available & ~has_ga & ~has_va',
    }
    stud_queries = {
        'Stud_Prim': 'current_edu == "pupil_primary"',
        'Stud_Sec_MobTool': 'current_edu == "pupil_secondary" & (car_available | has_ga | has_va)',
        'Stud_Sec_NoMobTool': 'current_edu == "pupil_secondary" & ~car_available & ~has_ga & ~has_va',
        'Stud_Ter_MobTool': 'current_edu == "student" & (car_available | has_ga | has_va)',
        'Stud_Ter_NoMobTool': 'current_edu == "student" & ~car_available & ~has_ga & ~has_va',
        'Stud_Appr_MobTool': 'current_edu == "apprentice" & (car_available | has_ga | has_va)',
        'Stud_Appr_NoMobTool': 'current_edu == "apprentice" & ~car_available & ~has_ga & ~has_va',
    }
    validate_person_groups(empl_queries)
    validate_person_groups(stud_queries)


def validate_persons(persons):
    """A few consistency assertions."""
    assert persons.query('~is_employed')['level_of_employment'].max() == 0, 'Issue with persons "is_employed"!'
    assert persons.query('is_employed')['level_of_employment'].min() > 0, 'Issue with persons "is_employed"!'
    assert persons.query('has_ga')['mobility'].isin(['car & ga', 'ga']).all(), 'Issue with persons "has_ga"!'
    assert persons.query('has_ht')['mobility'].isin(['car & va & ht', 'car & ht', 'ht', 'va & ht']).all(), 'Issue with persons "has_ht"!'
    assert persons.query('has_va')['mobility'].isin(['car & va & ht', 'car & va', 'va', 'va & ht']).all(), 'Issue with persons "has_va"!'
    assert persons.query('car_available')['mobility'].isin(['car & va & ht', 'car & va', 'car & ht', 'car & ga', 'car']).all(), 'Issue with persons "car_available"!'
    assert ~persons.query('mobility == "nothing"')[['has_ht', 'has_va', 'has_ga', 'car_available']].any().any(), 'Issue with persons "mobility"!'
    assert (persons['age'] >= 0).all(), 'Issue with persons "age"!'
    assert persons.query('car_available')['age'].min() == 18, 'Issue with persons "car_available"!'
    assert not (persons['level_of_employment'] > 100).any(), 'Some people have "level_of_employment" larger than 100!'
