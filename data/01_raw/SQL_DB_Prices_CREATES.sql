USE [Prices]
GO
/****** Object:  Table [dbo].[countries]    Script Date: 12/09/2020 12:16:20 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[countries](
	[Country_name] [nvarchar](80) NOT NULL,
	[ISO2] [char](2) NOT NULL,
	[ISO3] [char](3) NOT NULL,
	[ccTLD] [char](2) NOT NULL,
	[Continent] [char](12) NOT NULL
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[formats]    Script Date: 12/09/2020 12:16:20 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[formats](
	[Format] [text] NULL,
	[Measure] [text] NULL,
	[Weight] [float] NULL
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[models]    Script Date: 12/09/2020 12:16:20 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[models](
	[Model] [nvarchar](30) NULL,
	[Product] [nvarchar](30) NULL,
	[Country] [nvarchar](30) NULL,
	[Trade_Country] [nvarchar](100) NULL,
	[Category] [nvarchar](100) NULL,
	[Concept] [nvarchar](100) NULL,
	[Result] [float] NULL,
	[Updated] [datetime] NULL
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[prices]    Script Date: 12/09/2020 12:16:20 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[prices](
	[ID] [int] IDENTITY(1,1) NOT NULL,
	[Product] [nvarchar](30) NULL,
	[Country] [nvarchar](30) NULL,
	[Region] [nvarchar](100) NULL,
	[Trade_Country] [nvarchar](100) NULL,
	[Category] [nvarchar](100) NULL,
	[Package] [nvarchar](100) NULL,
	[Campaign] [int] NULL,
	[Campaign_wk] [int] NULL,
	[Date_price] [date] NULL,
	[Currency] [nvarchar](3) NULL,
	[Measure] [nvarchar](50) NULL,
	[Price] [float] NULL,
	[Updated] [datetime] NULL,
PRIMARY KEY CLUSTERED 
(
	[ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[prices_prediction]    Script Date: 12/09/2020 12:16:20 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[prices_prediction](
	[ID] [int] IDENTITY(1,1) NOT NULL,
	[Product] [nvarchar](30) NULL,
	[Country] [nvarchar](30) NULL,
	[Region] [nvarchar](100) NULL,
	[Trade_Country] [nvarchar](100) NULL,
	[Category] [nvarchar](100) NULL,
	[Package] [nvarchar](100) NULL,
	[Campaign] [int] NULL,
	[Campaign_wk] [int] NULL,
	[Date_price] [date] NULL,
	[Currency] [nvarchar](3) NULL,
	[Measure] [nvarchar](50) NULL,
	[Model] [nvarchar](50) NULL,
	[Model_measure] [nvarchar](50) NULL,
	[Model_measure_result] [float] NULL,
	[Price] [float] NULL,
	[Price_estimated] [float] NULL,
	[Updated] [datetime] NULL,
PRIMARY KEY CLUSTERED 
(
	[ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[regions]    Script Date: 12/09/2020 12:16:20 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[regions](
	[ID] [int] IDENTITY(1,1) NOT NULL,
	[Country] [nvarchar](30) NULL,
	[Region] [nvarchar](100) NULL,
	[State] [nvarchar](100) NULL,
	[Lat] [decimal](18, 8) NULL,
	[Lon] [decimal](18, 8) NULL,
PRIMARY KEY CLUSTERED 
(
	[ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[volumes]    Script Date: 12/09/2020 12:16:20 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[volumes](
	[ID] [int] IDENTITY(1,1) NOT NULL,
	[Product] [nvarchar](30) NULL,
	[Country] [nvarchar](30) NULL,
	[Region] [nvarchar](100) NULL,
	[Trade_Type] [nvarchar](10) NULL,
	[Trade_Country] [nvarchar](100) NULL,
	[Category] [nvarchar](100) NULL,
	[Package] [nvarchar](100) NULL,
	[Transport] [nvarchar](10) NULL,
	[Campaign] [int] NULL,
	[Campaign_wk] [int] NULL,
	[Date_volume] [date] NULL,
	[Measure] [nvarchar](50) NULL,
	[Volume] [float] NULL,
	[Updated] [datetime] NULL,
PRIMARY KEY CLUSTERED 
(
	[ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[volumes_prices]    Script Date: 12/09/2020 12:16:20 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[volumes_prices](
	[ID] [int] IDENTITY(1,1) NOT NULL,
	[Product] [nvarchar](30) NULL,
	[Country] [nvarchar](30) NULL,
	[Trade_Country] [nvarchar](100) NULL,
	[Category] [nvarchar](100) NULL,
	[Campaign] [int] NULL,
	[Campaign_wk] [int] NULL,
	[Date_ref] [date] NULL,
	[Currency] [nvarchar](3) NULL,
	[Measure] [nvarchar](50) NULL,
	[Price] [float] NULL,
	[Volume] [float] NULL,
	[Updated] [datetime] NULL,
PRIMARY KEY CLUSTERED 
(
	[ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]
GO
