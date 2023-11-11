import streamlit as st
import json
from PIL import Image
import manga
from manga import add_text_to_panel, generate_text_image, create_strip, text_to_image, generate_panels
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import nltk
from nltk.tokenize import sent_tokenize
import re
nltk.download('punkt')
model = GPT2LMHeadModel.from_pretrained("./modeltune")
tokenizer = GPT2Tokenizer.from_pretrained("./modeltune")

# Define the genre options
genre_options = ["Action", "Drama", "Comedy", "Romance"]
# Define the Streamlit app
st.title("Manga Comic Generator")

# Define the two tabs
tabs = ["Storyboard", "Get Inspired"]
selected_tab = st.selectbox("Select a tab:", tabs)

if selected_tab == "Storyboard":
    st.header("Create Your Manga Storyboard")
    st.markdown("This tool allows you to generate a manga panel (in a 4-koma style) for any manga plot ideas you have! A few things to keep in mind while entering your scenarios are listed below.")

    # Provide guidelines
    st.markdown("## Storyboard Guidelines")
    st.markdown("1. The generator takes about 3-5 minutes to work. Good things come to those who wait.")
    st.markdown("2. The more descriptive the scenario, the better. Feel free to describe your characters and the journey they go on, including locations.")
    st.markdown('3. A sample prompt could look like this:')
    st.markdown('"Characters: Yuusha is a brave male hero with blonde hair He is young, adventurous and daring. Nakama is his best friend and mentor who is a grizzled old adventurer. He is wise, old, and cautious. Yuusha and Nakama out on an adventure to defeat the evil demon lord. They fight many battles on their journey, including fighting a dragon. They traverse through harsh desert sandstorms, lush forest environments, and navigate treacherous dungeons. In the end, they take on the demon lord in an epic fight to the end together and win."')
    st.markdown("4. It is important to remember that this tool is meant as a way to help speed up storyboarding. The pictures generated are not, and can never be a substitute for actual art.")


    # User input for the scenario
    scenario = st.text_area("Enter your scenario:")

    if st.button("Generate Manga Panels"):
        if scenario:
            # Generate panels
            panels = generate_panels(scenario)

            # Initialize a list to store panel images
            panel_images = []

            # Generate images for each panel and add text
            for panel in panels:
               panel_prompt = panel["description"] + ", cartoon box, " + STYLE
               print(f"Generate panel {panel['number']} with prompt: {panel_prompt}")
               panel_image = text_to_image(panel_prompt)
               panel_image_with_text = add_text_to_panel(panel, panel_image)  # Pass the panel dictionary

            # Create a strip from the panel images and save it
            strip_image = create_strip(panel_images)

            # Display generated strip
            st.image(strip_image, use_column_width=True, caption="Generated Manga Comic Strip")
            
elif selected_tab == "Get Inspired":
    st.header("Let us help you come up with ideas!")
    st.markdown("This tool allows you to select a genre from the available list and generate an idea for a manga story! You can try entering the description in the storyboarding tool to generate a manga panel!!")

    # Add user input section
    user_genre = st.selectbox("Select a genre:", genre_options)

    if st.button("Generate Description"):
        # Set max_length to control the length of the generated text
        input_text = f"A {user_genre} manga:"
        input_ids = tokenizer.encode(input_text, return_tensors="pt")

        # Set attention_mask and pad_token_id
        attention_mask = input_ids.clone()
        attention_mask[attention_mask != tokenizer.pad_token_id] = 1  # Set non-pad tokens to 1
        pad_token_id = tokenizer.eos_token_id

        # Generate text using the model
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=100,  # Adjust max_length as needed
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            pad_token_id=pad_token_id,
        )

        # Decode the generated text
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        # Clean the generated text
        pattern = r'\(.*?\)'  # Matches anything within parentheses
        cleaned_text = re.sub(pattern, '', generated_text)
        sentences = sent_tokenize(cleaned_text)
        cleaned_sentences = sentences[:3]
        cleaned_text = " ".join(cleaned_sentences)

        st.subheader("Generated Manga Description:")
        # Display the cleaned text in green color
        st.markdown(f'<p style="color: green;">{cleaned_text}</p>', unsafe_allow_html=True)

        # Display genre-specific content below the description
        if user_genre == "Action":
            st.write("We are still working on getting this description for this genre spot on.. In the meanwhile, check out these 3 Action manga that are sure to get your creative juices flowing!")
            # Display Action manga recommendations with online image URLs and title above the image in bold
            manga_title = "**1. Chainsaw Man**"
            st.markdown(manga_title)  # Display manga title in bold above the image
            st.image("https://cdn.myanimelist.net/images/manga/3/216464.jpg", caption="Chainsaw Man")
            st.write("**Synopsis:** Denji has a simple dream—to live a happy and peaceful life, spending time with a girl he likes. This is a far cry from reality, however, as Denji is forced by the yakuza into killing devils in order to pay off his crushing debts. Using his pet devil Pochita as a weapon, he is ready to do anything for a bit of cash. Unfortunately, he has outlived his usefulness and is murdered by a devil in contract with the yakuza. However, in an unexpected turn of events, Pochita merges with Denji's dead body and grants him the powers of a chainsaw devil. Now able to transform parts of his body into chainsaws, a revived Denji uses his new abilities to quickly and brutally dispatch his enemies. Catching the eye of the official devil hunters who arrive at the scene, he is offered work at the Public Safety Bureau as one of them. Now with the means to face even the toughest of enemies, Denji will stop at nothing to achieve his simple teenage dreams.")
            # Display Action manga recommendations with online image URLs and title above the image in bold
            manga_title = "**2. Berserk**"
            st.markdown(manga_title)  # Display manga title in bold above the image
            st.image("https://cdn.myanimelist.net/images/manga/1/157897.jpg", caption="Berserk")
            st.write("**Synopsis:** Guts, a former mercenary now known as the 'Black Swordsman,' is out for revenge. After a tumultuous childhood, he finally finds someone he respects and believes he can trust, only to have everything fall apart when this person takes away everything important to Guts for the purpose of fulfilling his own desires. Now marked for death, Guts becomes condemned to a fate in which he is relentlessly pursued by demonic beings. Setting out on a dreadful quest riddled with misfortune, Guts, armed with a massive sword and monstrous strength, will let nothing stop him, not even death itself, until he is finally able to take the head of the one who stripped him—and his loved one—of their humanity.")
            # Display Action manga recommendations with online image URLs and title above the image in bold
            manga_title = "**3. Attack on Titan**"
            st.markdown(manga_title)  # Display manga title in bold above the image
            st.image("https://cdn.myanimelist.net/images/manga/2/37846.jpg", caption="Attack on Titan")
            st.write("**Synopsis:** Hundreds of years ago, horrifying creatures which resembled humans appeared. These mindless, towering giants, called 'Titans,' proved to be an existential threat, as they preyed on whatever humans they could find in order to satisfy a seemingly unending appetite. Unable to effectively combat the Titans, mankind was forced to barricade themselves within large walls surrounding what may very well be humanity's last safe haven in the world. In the present day, life within the walls has finally found peace, since the residents have not dealt with Titans for many years. Eren Yeager, Mikasa Ackerman, and Armin Arlert are three young children who dream of experiencing all that the world has to offer, having grown up hearing stories of the wonders beyond the walls. But when the state of tranquility is suddenly shattered by the attack of a massive 60-meter Titan, they quickly learn just how cruel the world can be. On that day, Eren makes a promise to himself that he will do whatever it takes to eradicate every single Titan off the face of the Earth, with the hope that one day, humanity will once again be able to live outside the walls without fear.")
        elif user_genre == "Comedy":
            st.write("We are still working on getting this description for this genre spot on.. In the meanwhile, check out these 3 Comedy manga that are sure to get your creative juices flowing!")
            # Display Action manga recommendations with online image URLs and title above the image in bold
            manga_title = "**1. Grand Blue**"
            st.markdown(manga_title)  # Display manga title in bold above the image
            st.image("https://cdn.myanimelist.net/images/manga/2/166124.jpg", caption="Grand Blue")
            st.write("**Synopsis:** Among the seaside town of Izu's ocean waves and rays of shining sun, Iori Kitahara is just beginning his freshman year at Izu University. As he moves into his uncle's scuba diving shop, Grand Blue, he eagerly anticipates his dream college life, filled with beautiful girls and good friends. But things don't exactly go according to plan. Upon entering the shop, he encounters a group of rowdy, naked upperclassmen, who immediately coerce him into participating in their alcoholic activities. Though unwilling at first, Iori quickly gives in and becomes the heart and soul of the party. Unfortunately, this earns him the scorn of his cousin, Chisa Kotegawa, who walks in at precisely the wrong time. Undeterred, Iori still vows to realize his ideal college life, but will things go according to plan this time, or will his situation take yet another dive?")
            # Display Action manga recommendations with online image URLs and title above the image in bold
            manga_title = "**2. Hinamatsuri**"
            st.markdown(manga_title)  # Display manga title in bold above the image
            st.image("https://cdn.myanimelist.net/images/manga/3/285804.jpg", caption="Hinamatsuri")
            st.write("**Synopsis:** Yoshifumi Nitta, a mid-level yakuza, finds his life forever changed when a strange capsule appears and rams into his head. Though he believes the curious event to be a dream, he finds the capsule still there the next morning; from within it emerges a young girl. She remembers nothing but her name, Hina, and uses psychokinesis to coerce Yoshifumi into buying her clothes and toys. Unable to get rid of Hina, Yoshifumi reluctantly becomes her guardian. The pair's peculiar life is just beginning. As Hina lazes around the house, Yoshifumi quickly rises through the ranks of the yakuza with the help of her supernatural abilities. Hinamatsuri follows the comedic duo as Hina drags both new friends and old acquaintances into her antics, while Yoshifumi juggles between taking care of Hina, hiding her powers, and managing the yakuza business.")
            # Display Action manga recommendations with online image URLs and title above the image in bold
            manga_title = "**3. Asobi Asobase**"
            st.markdown(manga_title)  # Display manga title in bold above the image
            st.image("https://cdn.myanimelist.net/images/manga/1/180830.jpg", caption="Asobi Asobase")
            st.write("**Synopsis:** Kasumi is a girl that hates playing games and is struggling with her low English grades. However, her fate is about to take a playful turn thanks to her colleagues: the airheaded Kasumi and the 'overseas' transfer student Olivia! Will these three girls play a lot of different games together? Absolutely yes! Will Kasumi English grades improve at all? Absolutely n... well, that remains to be seen.")
        elif user_genre == "Drama":
            st.write("We hope that description has helped spark your creativity! If you are looking for more inspiration, check out the following Drama manga that are sure to get your creative juices flowing!")
            # Display Action manga recommendations with online image URLs and title above the image in bold
            manga_title = "**1. Oyasumi Punpun**"
            st.markdown(manga_title)  # Display manga title in bold above the image
            st.image("https://cdn.myanimelist.net/images/manga/3/266834.jpg", caption="Oyasumi Punpun")
            st.write("**Synopsis:** Punpun Onodera is a normal 11-year-old boy living in Japan. Hopelessly idealistic and romantic, Punpun begins to see his life take a subtle—though nonetheless startling—turn to the adult when he meets the new girl in his class, Aiko Tanaka. It is then that the quiet boy learns just how fickle maintaining a relationship can be, and the surmounting difficulties of transitioning from a naïve boyhood to a convoluted adulthood. When his father assaults his mother one night, Punpun realizes another thing: those whom he looked up to were not as impressive as he once thought. As his problems increase, Punpun's once shy demeanor turns into voluntary reclusiveness. Rather than curing him of his problems and conflicting emotions, this merely intensifies them, sending him down the dark path of maturity in this grim coming-of-age saga.")
            # Display Action manga recommendations with online image URLs and title above the image in bold
            manga_title = "**2. A Silent Voice**"
            st.markdown(manga_title)  # Display manga title in bold above the image
            st.image("https://cdn.myanimelist.net/images/manga/1/120529.jpg", caption="A Silent Voice")
            st.write("**Synopsis:** Shouya Ishida, a mischievous elementary school student, finds himself troubled by deaf transfer student Shouko Nishimiya. Despite her genuine attempts to befriend her new classmates, Shouko only proves herself to be an annoyance for Shouya and his friends, provoking them to ridicule her at any possible chance. Soon enough, their taunts turn into constant assault—yet the teachers heartlessly remain apathetic to the situation. Shouya's misdeeds are finally stopped when Shouko transfers to another school. Denying their involvement, the entire class puts the blame on Shouya. As the new victim of bullying, Shouya gradually becomes meek and reclusive, being treated with contempt and disregard for years to come. Now a high school student, Shouya meets Shouko again for the first time in five years. Still tormented by his past actions, Shouya is determined to make amends for his mistakes and confront his trauma—even if he must face arduous obstacles along the way.")
            # Display Action manga recommendations with online image URLs and title above the image in bold
            manga_title = "**3. Monster**"
            st.markdown(manga_title)  # Display manga title in bold above the image
            st.image("https://cdn.myanimelist.net/images/manga/3/258224.jpg", caption="Monster")
            st.write("**Synopsis:** Kenzou Tenma, a renowned Japanese neurosurgeon working in post-war Germany, faces a difficult choice: to operate on Johan Liebert, an orphan boy on the verge of death, or on the mayor of Düsseldorf. In the end, Tenma decides to gamble his reputation by saving Johan, effectively leaving the mayor for dead. As a consequence of his actions, hospital director Heinemann strips Tenma of his position, and Heinemann's daughter Eva breaks off their engagement. Disgraced and shunned by his colleagues, Tenma loses all hope of a successful career—that is, until the mysterious killing of Heinemann gives him another chance. Nine years later, Tenma is the head of the surgical department and close to becoming the director himself. Although all seems well for him at first, he soon becomes entangled in a chain of gruesome murders that have taken place throughout Germany. The culprit is a monster—the same one that Tenma saved on that fateful day nine years ago.")
        elif user_genre == "Romance":
            st.write("We hope that description has helped spark your creativity! If you are looking for more inspiration, check out the following Romance manga that are sure to get your creative juices flowing!")
            # Display Action manga recommendations with online image URLs and title above the image in bold
            manga_title = "**1. I sold my life for ten thousand yen per year.**"
            st.markdown(manga_title)  # Display manga title in bold above the image
            st.image("https://cdn.myanimelist.net/images/manga/5/260043.jpg", caption="I sold my life for ten thousand yen per year.")
            st.write("**Synopsis:** Helpless and struggling for cash, 20-year-old Kusunoki sells the last of his possessions to buy food. Noticing his poverty, an old shop owner directs him to a store that supposedly purchases lifespan, time, and health. While not completely believing the man's words, Kusunoki nevertheless finds himself at the address out of desperation and curiosity. Kusunoki is crushed when he finds out the true monetary value of his lifespan—totaling a meager three hundred thousand yen. Deciding to sell the next 30 years of his life for ten thousand yen per year, Kusunoki is left with only three months to live. After heading home with the money, he is greeted by an unexpected visitor: the same store clerk he sold his lifespan to. She introduces herself as Miyagi, the one tasked with the job of observing him until the last three days of his life. Jumyou wo Kaitotte Moratta. Ichinen ni Tsuki, Ichimanen de. follows the remaining three months of Kusunoki's life as he confronts lingering regrets from the past and discovers what truly gives life value.")
            # Display Action manga recommendations with online image URLs and title above the image in bold
            manga_title = "**2. Fruits Basket**"
            st.markdown(manga_title)  # Display manga title in bold above the image
            st.image("https://cdn.myanimelist.net/images/manga/3/269697.jpg", caption="Fruits Basket")
            st.write("**Synopsis:** Tooru Honda is an orphan with nowhere to go but a tent in the woods, until the Souma family takes her in. However, the Souma family is no ordinary family, and they hide a grave secret: when they are hugged by someone of the opposite gender, they turn into animals from the Chinese zodiac! Now, Tooru must help Kyou and Yuki Souma hide their curse from their classmates, as well as her friends Arisa Uotani and Saki Hanajima. As she is drawn further into the mysterious world of the Soumas, she meets more of the family, forging friendships along the way. But this curse has caused much suffering; it has broken many Soumas. Despite this, Tooru may just be able to heal their hearts and soothe their souls.")
            # Display Action manga recommendations with online image URLs and title above the image in bold
            manga_title = "**3. Bloom Into You**"
            st.markdown(manga_title)  # Display manga title in bold above the image
            st.image("https://cdn.myanimelist.net/images/anime/1783/96153.jpg", caption="Bloom Into You")
            st.write("**Synopsis:** Yuu Koito has always enjoyed romance manga and love songs. She clings to them with the hope that she will one day experience a love story of her own—one that will sweep her off her feet and make her heart flutter. However, reality is often disappointing. When a classmate from junior high confesses his feelings to her, Yuu finds that she feels nothing. Unable to give him an answer, she becomes convinced that she is unable to fall in love. One day, on her way to the student council room, Yuu encounters the council's president, Touko Nanami, turning down a confession from a boy. Inspired by Touko's confidence, Yuu turns to her for help. But when Touko becomes the next person to confess to Yuu, she is confused, yet her heart is set aflutter.")
            