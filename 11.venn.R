library("ggVennDiagram")
library("ggvenn")

ukb_known = read.csv('ukb_known_multimorbidity.csv')
ukb_novel = read.csv('ukb_novel_multimorbidity.csv')
hudine_known = read.csv('hidalgo_known_multimorbidity.csv')
hudine_novel = read.csv('hidalgo_novel_multimorbidity.csv')


ukb_known = as.character(ukb_known$code)
ukb_novel = as.character(ukb_novel$code)
hudine_known = as.character(hudine_known$code)
hudine_novel = as.character(hudine_novel$code)


multimorbiidty = list(UKB_known=ukb_known, UKB_novel=ukb_novel, HuDiNe_novel=hudine_novel, HuDiNe_known=hudine_known)

fig = ggvenn(
  multimorbiidty, 
  fill_color = c("#0073C2FF", "#EFC000FF", "#868686FF", "#CD534CFF"),
  stroke_size = 0.5, set_name_size = 10, show_percentage = FALSE, text_size = 10
)

ggsave(fig, file='fig4_c.pdf', width=12, height=8)


library(extrafont)
loadfonts(device = "pdf")


