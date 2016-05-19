import json
from pprint import pprint

with open('profiles.json', 'r') as profile_data:
    data = json.load(profile_data)

for entry in data:

    for key in entry:

        # '_id'
        # 'active'
        # 'name'
        # 'firstname'
        # 'lastname'
        # 'emailaddress'
        # 'phone'
        # 'gender'
        # 'scope'
        # 'picture'
        # 'picturesource'
        # 'enabled'

        # 'facebook'
        # Facebook Items
        if str(key) == 'facebook':
            for fb_entry in entry[key]:
                # pprint(entry[key][fb_entry])

                # 'verified'

                # 'middlename'

                # 'firstname'

                # 'languages' [
                #               'id'
                #               'name'
                #               ]
                # if str(fb_entry) == 'languages':
                #     for lang_fb_entry in entry[key][fb_entry]:
                #         for fb_entry_lang_key in lang_fb_entry:
                #             pprint(lang_fb_entry[fb_entry_lang_key])

                # 'education' []
                if str(fb_entry) == 'education':
                    for education_fb_entry in entry[key][fb_entry]:
                        # pprint(education_fb_entry)
                        for education_fb_entry_key in education_fb_entry:
                            if len(education_fb_entry[education_fb_entry_key]) > 0:
                                for d in education_fb_entry[education_fb_entry_key]:
                                    pprint(education_fb_entry[education_fb_entry_key])
                                    pprint(d)
                                    exit()
                        exit()

                # 'name'
                # 'gender'
                # 'error'
                # 'updatedtime'
                # 'location' []
                # 'work' []
                # 'timezone'
                # 'locale'
                # '_id' []
                # 'emailaddress'
                # 'lastname'
                # 'id'
                # 'birthday'
                # 'relationshipstatus'

        # 'registered'
        # Registered items (usec and sec)
        # if str(key) == 'registered':
        #     for registered_entry in entry[key]:
        #         pprint(registered_entry)

        # 'birthday'
        # Birthday items (usec and sec)
        # if str(key) == 'birthday':
        #     for birthday_entry in entry[key]:
        #         pprint(birthday_entry)

        # 'social'
        # Social items
        # if str(key) == 'social':
        #     for social_entry in entry[key]:
        #         pprint(social_entry)

    #exit()
