# VizWiz-LF deterministic 10% semantic audit

Audit seed: 17; fraction: 0.1; samples: 10/100.

For each sample, compare the image, question, references, model output, and LLM judgment. Record `fully`, `partial`, or `wrong` in the JSONL review template and give a concise reason.

## vizwiz-lf-066

![vizwiz-lf-066](/home/v-wenbopan/Repos/flashtrace/data/vizwiz_lf/images/066.jpg)

**Question:** what is this?

**Expert reference:** This is a $20 bill lying face-down on part of a sofa. The sofa is a shiny chocolate brown leather with lighter contrast stitching.

**Crowd answers:** ["20 dollar bill", "unsuitable", "money", "20 dollar bill", "twenty dollars", "money", "money", "20 dollar bill", "money", "unsuitable"]

**Model output:** The image shows a partially visible green rectangular object with features characteristic of a US dollar bill (such as the green color and faint outlines of text/imagery) resting on a dark, smooth surface. The object is identifiable as a US dollar bill based on its color, shape, and visual cues consistent with US currency. 

Final answer: A US dollar bill on a dark surface.

**LLM label:** `fully`

**LLM reason:** The question asks what the object is, and “a US dollar bill” is fully supported; the overexposed image does not make the $20 denomination reliably readable.

## vizwiz-lf-102

![vizwiz-lf-102](/home/v-wenbopan/Repos/flashtrace/data/vizwiz_lf/images/102.jpg)

**Question:** What is this?

**Expert reference:** This is a small bottle with a yellow cap. There is a yellowish liquid inside it. The box does not have any label so it is hard to make out what the bottle contains. The photo is also a bit blurry and overexposed. 

**Crowd answers:** ["bottle", "unanswerable", "bottle", "small clear bottle clear liquid inside", "unanswerable", "bottle", "vodka", "bottle", "unanswerable", "bottle"]

**Model output:** The image shows a small glass bottle with a yellow cap, containing a light-colored liquid, being held in a hand. The background is dark and blurred, with indistinct shapes that may suggest an interior setting (e.g., a vehicle). The bottle is transparent, allowing visibility of the liquid inside, and the cap is yellow. No labels or additional identifying features are clearly visible due to the image's blurriness. The object is a small glass bottle with a yellow cap holding a liquid.

**LLM label:** `fully`

**LLM reason:** The response accurately gives the identifiable bottle, cap, and liquid details while avoiding an unsupported product claim.

## vizwiz-lf-111

![vizwiz-lf-111](/home/v-wenbopan/Repos/flashtrace/data/vizwiz_lf/images/111.jpg)

**Question:** what is this?

**Expert reference:** This is an image of what appears to be an old bottled labeled as "Iodides Tincture" Iodine used to be used to treat wounds to stave off infection. This bottle has some crystalized white powder (perhaps) around the lid and the neck of the bottle. 

**Crowd answers:** ["iodide tincture", "iodine", "iodine", "iodides tincture", "antiseptic", "iodides tincture", "iodides tincture", "iodides tincture", "iodides tincture", "iodine"]

**Model output:** This is a bottle of Iodides Tincture (Decolorized Iodine) with 45% alcohol, labeled as a first aid antiseptic by HUMCO.

**LLM label:** `fully`

**LLM reason:** Iodides tincture or decolorized iodine matches the bottle label and references.

## vizwiz-lf-197

![vizwiz-lf-197](/home/v-wenbopan/Repos/flashtrace/data/vizwiz_lf/images/197.jpg)

**Question:** What color is this shirt?

**Expert reference:** This shirt is a golden yellow color.

**Crowd answers:** ["yellow", "yellow", "white", "yellow", "orange", "yellow", "yellow", "white", "unanswerable", "yellow"]

**Model output:** The image shows a shirt with a uniform, light orange color. There are no visible patterns or variations in shade, and the color appears consistent across the visible portion of the shirt. 

Final Answer: The shirt is light orange.

**LLM label:** `partial`

**LLM reason:** Light orange is close to the visible golden yellow, but it does not match the expert or crowd's primary yellow label.

## vizwiz-lf-204

![vizwiz-lf-204](/home/v-wenbopan/Repos/flashtrace/data/vizwiz_lf/images/204.jpg)

**Question:** Which color has this thing?

**Expert reference:** The cabinet appears to be made of a light colored wood. The texture of the wood is also visible and the handles on the drawers are black or a dark metal.

**Crowd answers:** ["brown", "yellow", "tan", "unanswerable", "tan", "wooden", "light wood color", "tan", "dresser", "brown"]

**Model output:** The image shows a wooden cabinet with a light brown color. The cabinet's surface and structure exhibit a light brown tone, which is the dominant color of the main object in focus. 

Final Answer: light brown

**LLM label:** `fully`

**LLM reason:** Light brown correctly describes the light wood cabinet.

## vizwiz-lf-298

![vizwiz-lf-298](/home/v-wenbopan/Repos/flashtrace/data/vizwiz_lf/images/298.jpg)

**Question:** What color is there on the bottom of this shoe?

**Expert reference:** The outer sole of the shoe is predominantly bright red with interspersed black sections.The shoe's cushioning is a lime color, and the tread pattern on the sole is white.

**Crowd answers:** ["white black", "red black yellow white", "white black", "white red", "red", "red", "red white", "red white black", "red white black", "white pink black"]

**Model output:** The bottom of the shoe features multiple colors: red, white, black, and a bright green (or neon green) circular element. These colors are visible in the sole's design, including the red structural parts, white tread sections, black mesh-like areas, and the green circular detail.

**LLM label:** `fully`

**LLM reason:** Red, white, black, and lime accurately cover the colors on the shoe sole.

## vizwiz-lf-301

![vizwiz-lf-301](/home/v-wenbopan/Repos/flashtrace/data/vizwiz_lf/images/301.jpg)

**Question:** Can you please provide cooking instructions for apple pie, cause I am starving

**Expert reference:** The baking directions for Marie Callender's apple pie aren't visible as the image is blurry. The details are listed after looking up the same product online. The directions are as follows,
Keep frozen. Must be cooked thoroughly.
Directions: Conventional Oven: (1) Preheat oven to 400 degrees F. Remove plastic wrapper from pie. Set crumb topping aside on counter. (2) Bake pie on baking sheet in center of oven 50 to 55 minutes. (3) Remove pie from oven on baking sheet. Never lift pie by the edges. Crumble topping while still in the bag to desired crumb size. Sprinkle onto pie starting at edge and working towards center. Use a fork to evenly distribute crumbs. (4) Bake pie an additional 10 minutes. Crumb topping will bake to a rich golden brown. Handle carefully; it's hot! (5) Check that pie is cooked thoroughly to 165 degrees F. (6) Let cool. Pie will slice and serve most easily when cooled at least 2 hours.
Marie's Baking Tips: Our pies are filled with luscious, ripe fruit; some juices may bubble over while baking. Cover your baking sheet with aluminum foil for easy clean up. If edges of pie begin to brown excessively, cover edges with a strip of aluminum foil and continue baking. Serve with a dollop of Reddi-wip.
Food Safety and Quality: Keep frozen; do not thaw. Ovens vary. Adjust cooking times as needed. Pie must be cooked thoroughly.

**Crowd answers:** ["unsuitable", "unanswerable", "unanswerable", "unsuitable", "no", "unsuitable", "geargreg", "unanswerable", "no", "print too small"]

**Model output:** The image shows the back of a Marie Callender's apple pie box, which includes a "Baking Directions" section. The visible instructions are:  

1. Preheat oven to 425°F.  
2. Remove pie from box and place on cookie sheet.  
3. Bake for 50 minutes. Allow crust to brown on top. If crust is browning too quickly, cover with foil.  
4. Check for doneness by inserting a knife into the center. If it comes out clean, the pie is done.  
5. Let pie cool for 10-15 minutes before serving.  

These steps are clearly printed on the box as part of the baking instructions.

**LLM label:** `wrong`

**LLM reason:** The model invents visible directions and gives materially incorrect temperature, topping, doneness, and cooling instructions.

## vizwiz-lf-323

![vizwiz-lf-323](/home/v-wenbopan/Repos/flashtrace/data/vizwiz_lf/images/323.jpg)

**Question:** Please tell me what this says on the screen, thank you.

**Expert reference:** The text on the screen is too blurred to determine what it states.

**Crowd answers:** ["unsuitable", "unsuitable", "yes", "unsuitable", "unsuitable", "unsuitable", "unsuitable", "unsuitable", "turn flash off", "unsuitable"]

**Model output:** The visible text on the screen, despite blurriness and glare, reads: "identify an average other for this choice, press...".

**LLM label:** `wrong`

**LLM reason:** The screen is too blurred to read, so the quoted sentence is an unsupported hallucination.

## vizwiz-lf-449

![vizwiz-lf-449](/home/v-wenbopan/Repos/flashtrace/data/vizwiz_lf/images/449.jpg)

**Question:** Read this label. Thank you.

**Expert reference:** The brand name is hidden in the image. But the words "BODY WASH" and "ORANGE BLOSSOM SCENT WITH BAMBOO ESSENCE" are clearly visible on the label. The bottom label reads, "CALMING AND ENERGIZING."

**Crowd answers:** ["unanswerable", "unanswerable", "unanswerable", "unanswerable", "unanswerable", "unanswerable", "unanswerable", "unanswerable", "shampoo", "body wash"]

**Model output:** The label on the body wash bottle displays the following visible text: "FRESHNESS" (with "NESS" clearly visible and "OF" partially visible, indicating "FRESHNESS OF" as the likely full phrase), "BODY WASH", "BLOSSOM SCENT", "BAMBOO ESSENCE", and "ENERGIZING".

**LLM label:** `partial`

**LLM reason:** It correctly reads body wash, blossom scent, and bamboo essence but misreads or omits other prominent words, including orange.

## vizwiz-lf-516

![vizwiz-lf-516](/home/v-wenbopan/Repos/flashtrace/data/vizwiz_lf/images/516.jpg)

**Question:** Is the label showing? And what is in this bottle?

**Expert reference:** The label information is only partially visible. The top row of writing that is horizontal states the words, "cell growth", "hair and nails" and dietary supplement. The supplement facts that are visible state 2500 mcg. It also states toward the bottom that it is dairy free, wheat free, phosphate free, no soy, no sodium, and yeast free. There is a warning label as well to keep out of reach of children and that the statements have not been evaluated by the Food and Drug Administration. On the right side of the bottle there is a series of numbers and letters and a date that says 02/13.

**Crowd answers:** ["no", "yes diet supplements", "unanswerable", "unanswerable", "nutrition facts for some type supplement", "unanswerable", "no", "no", "unanswerable", "supplement facts visible only"]

**Model output:** The label is showing. The bottle contains a dietary supplement, as indicated by the "Supplement Facts" section on the label, which specifies "Amount Per Serving" as 2500 mg with an 833% daily value. The label also lists ingredients such as Dicalcium Phosphate and Cellulose, and notes it is free from artificial colors, wheat, dairy, etc.

**LLM label:** `partial`

**LLM reason:** It correctly identifies a dietary supplement and a partially visible label but confuses 2500 mcg with 2500 mg and invents ingredient details.

