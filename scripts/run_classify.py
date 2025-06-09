from classify_vibes import process_video

result = process_video(
    video_id="2025-05-28_13-42-32_UTC",
    caption="Ofcourse I'll get you flowers 🙆🏻‍♀️🙂‍↕️ Spinning into summer with my favorite @virgio.official dress, you like it too? I got you girlie, comment 'Link' and I will slide into your dms with the link 🤜🤛 Use code 'SUKRUTIAIRI' and save some extra 💸 Location- @roasterycoffeehouseindia 📍Noida",
    hashtags=["#grwm", "#summer", "#summerfit", "#dress", "#date", "#datedress", "#outfit", "#fashion", "#outﬁtinspo"],
    output_file="outputs/vibes_2025-05-28_13-42-32_UTC.json",
    threshold=0.4,
    top_k=3,
    caption_weight=0.8,
    hashtag_weight=0.2,
    keyword_weight=0.4,
    debug=True
)

print("Classification result:", result) 