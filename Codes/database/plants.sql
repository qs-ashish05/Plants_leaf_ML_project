use mysql;

create table disease(
predicted_disease varchar(500),
cause varchar(500),
prevention varchar(10000)
);

insert into disease values 
("Pepper__bell___Bacterial_spot","Xanthomonas campestris pv. vesicatoria","Seed treatment with hot water, soaking seeds for 30 minutes in water pre-heated to 125 F/51 C");

insert into disease values
("Pepper__bell___healthy","Your plant is  healthy","No need of any prevention");

insert into disease values
("Tomato_Early_blight","Fungus-Alternaria solani","spray the plant with liquid copper fungicide concentrate");

insert into disease values
("Tomato_Late_blight","Fungus-Phytophthora infestans","Pull out the plant from the garden as soon as possible");

insert into disease values
("Tomato_healthy","Your plant is  healthy","No need of any prevention");

select * from disease