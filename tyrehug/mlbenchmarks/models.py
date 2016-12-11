from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

engine = create_engine('sqlite:///:memory:', echo=True)
Base = declarative_base()


class Dataset(Base):
    __tablename__ = 'dataset'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    rows = Column(Integer)
    cols = Column(Integer)

    def __repr__(self):
        return "<Dataset(name='%s', rows='%s', cols='%s')>" % (self.name, self.rows, self.cols)
